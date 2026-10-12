"""
score_customer.py — Build features for one or more customer_ids through the same
features.build.build_features() entry point the training script uses.

Versioned features are scored with the parameters of the feature store's *latest* version
(the version a model trained from the store would have seen), not build_features' defaults
— otherwise training on txn_count_window v2 (30 days) and scoring at 90 days would be a
silent train/score skew. --params overrides this explicitly.

    python scripts/score_customer.py --customer-id 2 17 --as-of 2026-08-01
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd

from feature_store import store
from features.build import build_features
from features.schema import VERSIONED_FEATURES


def latest_store_params() -> dict[str, dict]:
    """Parameters of each versioned feature's is_latest registry row."""
    return store.latest_parameters([vf.name for vf in VERSIONED_FEATURES])


def main(argv=None):
    os.chdir(ROOT)  # data/ and feature_store/ paths are repo-relative
    parser = argparse.ArgumentParser(description="Build features for specific customers (scoring path)")
    parser.add_argument("--customer-id", type=int, nargs="+", required=True)
    # No default: feature_spec.md says scoring uses the date passed in; only the training
    # script carries a default as_of.
    parser.add_argument("--as-of", required=True, help="As-of date (ISO); use today's date for live scoring")
    parser.add_argument("--params", default=None, help='JSON override, e.g. \'{"txn_count_window": {"window_days": 90}}\'')
    args = parser.parse_args(argv)

    params = json.loads(args.params) if args.params else latest_store_params()
    unpinned = [vf.name for vf in VERSIONED_FEATURES if vf.name not in params]
    if unpinned:
        raise SystemExit(
            f"No registered version for {unpinned}: run scripts/build_training_features.py or pass --params, "
            "rather than silently scoring with build_features' first-version defaults."
        )
    as_of = pd.Timestamp(args.as_of)
    df = build_features(as_of=as_of, customer_ids=args.customer_id, params=params)

    missing = sorted(set(args.customer_id) - set(df["customer_id"]))
    if missing:
        print(f"Not scored (unknown customer_id or signed up on/after as_of): {missing}")
    print(f"as_of={as_of.date()}  versioned-feature params={params}")
    print(df.set_index("customer_id").T.to_string())


if __name__ == "__main__":
    main()
