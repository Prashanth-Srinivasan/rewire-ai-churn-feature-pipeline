"""
build_training_features.py — Build features for the full training population and register
every candidate feature in the feature store, including the spec's versioning exercise.

Calls the shared features.build.build_features() with customer_ids=None (everyone).

Registration is idempotent on top of whatever the store already holds (the store is never
reset): for each (feature, parameters) the script looks for an existing version with the
same parameters; if its stored values are identical, nothing is registered. If the values
differ, a new version is registered — but only when REVISION_NOTES states why the
computation changed, so no version is ever created without a stated reason.

    python scripts/build_training_features.py [--as-of 2026-05-03]

The training as_of default lives here and only here (feature_spec.md, parity design):
2026-05-03 is 90 days before 2026-08-01, the date the churned label appears to be defined
against, so training features use only data from before the label window.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from feature_store import store
from features import ingest
from features.build import DEFAULT_SOURCE_PATHS, build_features
from features.quality_checks import assert_all_checks_pass, zero_activity_ids_from_raw
from features.schema import FEATURES, VERSIONED_FEATURES, FeatureSpec, get_feature_spec

TRAINING_AS_OF = "2026-05-03"

# Why a feature's values changed while its parameters did not. Required for any such
# re-registration; keyed by feature name, applies to the version registered in this run.
# Empty since the store was rebuilt from scratch at 2026-05-03: every feature's v1 already
# carries the corrected computation (see ARTIFACT_NOTES.md for the earlier spend_ratio fix).
REVISION_NOTES: dict[str, str] = {}


def _params_key(params: dict) -> str:
    return json.dumps(params or {}, sort_keys=True)


def _same_values(stored: pd.DataFrame, new: pd.DataFrame, feature: str, as_of: pd.Timestamp) -> bool:
    a = stored[["customer_id", feature]].sort_values("customer_id").reset_index(drop=True)
    b = new[["customer_id", feature]].sort_values("customer_id").reset_index(drop=True)
    if len(a) != len(b) or not (a["customer_id"].values == b["customer_id"].values).all():
        return False
    if a[feature].dtype == object or b[feature].dtype == object or str(b[feature].dtype) == "str":
        same = (a[feature].astype(str).values == b[feature].astype(str).values).all()
    else:
        same = np.allclose(a[feature].astype(float), b[feature].astype(float), rtol=0, atol=1e-12)
    # Snapshots registered before as_of_date was enforced store it as null; those are
    # compared on values only (documented in ARTIFACT_NOTES.md).
    stored_as_of = stored["as_of_date"].dropna().unique()
    return bool(same) and all(pd.Timestamp(x) == as_of for x in stored_as_of)


def register_if_changed(
    spec: FeatureSpec,
    features_df: pd.DataFrame,
    as_of: pd.Timestamp,
    params: dict,
    rationale: str,
) -> str:
    feature = spec.name
    values = features_df[["customer_id", feature]].copy()
    values["as_of_date"] = as_of.date().isoformat()

    existing = store.list_versions(feature)
    same_params = existing[existing["parameters"].map(lambda p: _params_key(json.loads(p))) == _params_key(params)]
    revision_note = None
    if len(same_params):
        prior = same_params.iloc[0]  # list_versions is most-recent first
        if _same_values(store.get_version(feature, int(prior["version"])), values, feature, as_of):
            return f"unchanged, v{int(prior['version'])} already holds these values"
        revision_note = REVISION_NOTES.get(feature)
        if revision_note is None:
            raise SystemExit(
                f"{feature} {params or ''}: values differ from stored v{int(prior['version'])} with the same "
                "parameters. Add a REVISION_NOTES entry explaining the computation change before re-registering."
            )

    param_text = f" Parameters: {json.dumps(params)}." if params else ""
    version = store.register_version(
        feature_name=feature,
        values_df=values,
        description=spec.description,
        source_columns="; ".join(spec.sources),
        transform_summary=(
            f"{spec.computation}.{param_text} Zero-activity default: {spec.zero_activity_default}. "
            f"Built as of {as_of.date()} (rows with txn_timestamp < as_of only)."
        ),
        business_rationale=f"{revision_note} {spec.rationale}" if revision_note else rationale,
        parameters=params,
    )
    return f"registered v{version}"


def main(argv=None):
    os.chdir(ROOT)  # data/ and feature_store/ paths are repo-relative
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default=TRAINING_AS_OF)
    as_of = pd.Timestamp(parser.parse_args(argv).as_of)

    print(f"Building training features (as_of={as_of.date()})...")
    df = build_features(as_of=as_of)
    print(f"  {len(df)} rows x {len(df.columns) - 1} features")

    customers = ingest.load_customers_csv(DEFAULT_SOURCE_PATHS["customers"])
    customers = customers[customers["signup_date"] < as_of]
    txns = ingest.load_transactions_sqlite(DEFAULT_SOURCE_PATHS["transactions_db"], as_of=as_of)
    zero_ids = zero_activity_ids_from_raw(txns, customers["customer_id"], as_of)
    signup_dates = dict(zip(customers["customer_id"], customers["signup_date"]))

    print("\nQuality checks (default parameters):")
    assert_all_checks_pass(df, zero_ids, len(customers), signup_dates, as_of)

    versioned = {vf.name for vf in VERSIONED_FEATURES}
    print("\nRegistering non-versioned candidate features:")
    for spec in FEATURES:
        if spec.name not in versioned:
            print(f"  {spec.name}: {register_if_changed(spec, df, as_of, {}, spec.rationale)}")

    print("\nVersioning exercise (feature_spec.md):")
    for vf in VERSIONED_FEATURES:
        spec = get_feature_spec(vf.name)
        for v in vf.versions:
            vdf = build_features(as_of=as_of, params={vf.name: v.parameters})
            print(f"  quality checks for {vf.name} {v.parameters}:")
            assert_all_checks_pass(vdf, zero_ids, len(customers), signup_dates, as_of)
            status = register_if_changed(spec, vdf, as_of, v.parameters, f"{v.reason} {spec.rationale}")
            print(f"  {vf.name} {v.parameters}: {status}")

        latest = store.list_versions(vf.name).iloc[0]
        if int(latest["is_latest"]) != 1 or json.loads(latest["parameters"]) != vf.versions[-1].parameters:
            raise SystemExit(f"{vf.name}: latest version does not carry the spec's final parameters")
        print(f"  {vf.name}: is_latest -> v{int(latest['version'])} {latest['parameters']}")

    print("\nBuild complete. Registry: feature_store/registry.db, snapshots: feature_store/values/")


if __name__ == "__main__":
    main()
