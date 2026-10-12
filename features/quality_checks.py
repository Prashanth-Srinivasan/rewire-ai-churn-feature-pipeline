"""
quality_checks.py — Null / default / bounds / row-count / drift checks on a features DataFrame.

Each check returns (passed: bool, message: str); the message names the offending
features and customer_ids. assert_all_checks_pass() raises SystemExit(1) on any failure —
a failing check never just logs a warning.

Runnable standalone:
    python -m features.quality_checks --as-of 2026-05-03            # spec checks
    python -m features.quality_checks --as-of 2026-05-03 --drift    # + drift vs feature store latest
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

import numpy as np
import pandas as pd

from features.schema import VERSIONED_FEATURES, get_feature_names


KNOWN_CHANNELS = frozenset({"organic", "paid_search", "referral", "social"})

# Bounds from feature_spec.md "Data-quality checks". None = unbounded on that side.
SPEC_BOUNDS: dict[str, tuple[Optional[float], Optional[float]]] = {
    "spend_ratio_last_3m_vs_previous": (0, 500),
    "txn_ratio_last_3m_vs_previous": (0, 100),
    "return_pct": (0, 100),
    "return_count_recent": (0, 10),
    "txn_count_window": (0, 50),
    "life_span_days": (0, 900),
    "days_since_last_txn": (0, None),
}

# Fixed zero-activity defaults from the spec. days_since_last_txn's default varies per
# customer (days from signup to as_of) and is checked separately.
ZERO_ACTIVITY_DEFAULTS = {
    "life_span_days": 0,
    "return_count_recent": 0,
    "spend_ratio_last_3m_vs_previous": 1.0,
    "txn_ratio_last_3m_vs_previous": 1.0,
    "has_txn_history": 0,
    "txn_count_window": 0,
    "return_pct": 0.0,
}


def _ids(df: pd.DataFrame, mask: pd.Series, limit: int = 10) -> list:
    return df.loc[mask, "customer_id"].head(limit).tolist()


def check_no_nulls(df: pd.DataFrame, feature_names: list[str]) -> tuple[bool, str]:
    """No nulls in any feature column after defaults are applied (and every column present)."""
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        return False, f"Feature columns missing from output: {missing}"
    failures = {c: _ids(df, df[c].isnull()) for c in feature_names if df[c].isnull().any()}
    if failures:
        return False, f"Nulls found (feature -> first customer_ids): {failures}"
    return True, "No nulls found in any feature column."


def check_zero_activity_defaults(
    df: pd.DataFrame,
    zero_activity_customer_ids: set,
    defaults: dict = ZERO_ACTIVITY_DEFAULTS,
) -> tuple[bool, str]:
    """Customers with no transactions before as_of must carry exactly the spec'd defaults."""
    mask = df["customer_id"].isin(zero_activity_customer_ids)
    failures = {}
    for feat, default in defaults.items():
        bad = mask & (df[feat] != default)
        if bad.any():
            failures[feat] = {"expected": default, "customer_ids": _ids(df, bad), "got": df.loc[bad, feat].head(5).tolist()}
    if failures:
        return False, f"Zero-activity default mismatches: {failures}"
    return True, f"All {int(mask.sum())} zero-activity customers carry the spec'd defaults."


def check_zero_activity_days_since_last_txn(
    df: pd.DataFrame,
    zero_activity_customer_ids: set,
    signup_dates: dict,
    as_of: pd.Timestamp,
) -> tuple[bool, str]:
    """Zero-activity customers' days_since_last_txn must equal days from signup_date to as_of."""
    as_of = pd.Timestamp(as_of)
    mask = df["customer_id"].isin(zero_activity_customer_ids)
    expected = df.loc[mask, "customer_id"].map(lambda c: (as_of - signup_dates[c]).days)
    bad = mask.copy()
    bad[mask] = df.loc[mask, "days_since_last_txn"] != expected
    if bad.any():
        return False, f"Zero-activity days_since_last_txn != days since signup for customer_ids {_ids(df, bad)}"
    return True, "Zero-activity days_since_last_txn equals days since signup."


def check_bounds(
    df: pd.DataFrame,
    bounds: dict[str, tuple[Optional[float], Optional[float]]] = SPEC_BOUNDS,
) -> tuple[bool, str]:
    """Every bounded feature stays within [lo, hi]."""
    failures = {}
    for feat, (lo, hi) in bounds.items():
        vals = df[feat]
        out = pd.Series(False, index=df.index)
        if lo is not None:
            out |= vals < lo
        if hi is not None:
            out |= vals > hi
        if out.any():
            failures[feat] = {"bounds": (lo, hi), "rows": int(out.sum()), "customer_ids": _ids(df, out)}
    if failures:
        return False, f"Bound violations: {failures}"
    return True, "All features within spec bounds."


def check_acquisition_channel(df: pd.DataFrame, known_channels=KNOWN_CHANNELS) -> tuple[bool, str]:
    """acquisition_channel is one of the four known values (missing counts as unknown)."""
    bad = ~df["acquisition_channel"].isin(known_channels)
    if bad.any():
        return False, f"Unknown acquisition channels {df.loc[bad, 'acquisition_channel'].unique().tolist()} for customer_ids {_ids(df, bad)}"
    return True, "All acquisition channels are known values."


def check_row_count(df: pd.DataFrame, expected_count: int) -> tuple[bool, str]:
    """Exactly one row per customer, and as many rows as the expected population."""
    dupes = df["customer_id"].duplicated(keep=False)
    if dupes.any():
        return False, f"Duplicate customer_id rows: {sorted(set(_ids(df, dupes, 20)))}"
    if len(df) != expected_count:
        return False, f"Expected {expected_count} rows (one per customer), got {len(df)}"
    return True, f"One row per customer; row count matches expected {expected_count}."


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI of current vs reference, binned on reference quantiles (categoricals: by value)."""
    eps = 1e-6
    if reference.dtype == object or str(reference.dtype) in ("str", "string", "category"):
        cats = sorted(set(reference) | set(current))
        ref_p = reference.value_counts(normalize=True).reindex(cats, fill_value=0).values
        cur_p = current.value_counts(normalize=True).reindex(cats, fill_value=0).values
    else:
        edges = np.unique(np.quantile(reference.astype(float), np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            edges = np.array([edges[0] - 0.5, edges[0] + 0.5])
        edges[0], edges[-1] = -np.inf, np.inf
        ref_p = np.histogram(reference.astype(float), edges)[0] / len(reference)
        cur_p = np.histogram(current.astype(float), edges)[0] / len(current)
    ref_p, cur_p = ref_p + eps, cur_p + eps
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def check_drift(
    current: pd.DataFrame,
    reference: dict[str, pd.Series],
    threshold: float = 0.25,
) -> tuple[bool, str]:
    """PSI of each feature vs a reference distribution; PSI > threshold (0.25 = 'major shift') fails.

    Not one of feature_spec.md's training-gate checks: it compares a new build against a
    previously registered snapshot, so it is run standalone (--drift), not in the training build.
    """
    scores = {f: population_stability_index(ref, current[f]) for f, ref in reference.items()}
    drifted = {f: round(s, 3) for f, s in scores.items() if s > threshold}
    if drifted:
        return False, f"Drift (PSI > {threshold}): {drifted}"
    return True, f"No drift above PSI {threshold} (max {max(scores.values(), default=0):.3f})."


def zero_activity_ids_from_raw(transactions: pd.DataFrame, customer_ids, as_of) -> set:
    """Customers with no raw transaction before as_of — computed independently of has_txn_history."""
    active = set(transactions.loc[transactions["txn_timestamp"] < pd.Timestamp(as_of), "customer_id"])
    return set(customer_ids) - active


def run_all_checks(
    df: pd.DataFrame,
    zero_activity_ids: set,
    expected_count: int,
    signup_dates: dict,
    as_of,
    bounds: dict = SPEC_BOUNDS,
) -> list[tuple[str, bool, str]]:
    """Run every feature_spec.md check; returns [(name, passed, message)]."""
    results = [("no_nulls", *check_no_nulls(df, get_feature_names())),
               ("row_count", *check_row_count(df, expected_count))]
    if any(c not in df.columns for c in get_feature_names()):
        return results  # the remaining checks need every column; no_nulls already reports which are missing
    results += [
        ("zero_activity_defaults", *check_zero_activity_defaults(df, zero_activity_ids)),
        ("zero_activity_days_since_last_txn",
         *check_zero_activity_days_since_last_txn(df, zero_activity_ids, signup_dates, as_of)),
        ("bounds", *check_bounds(df, bounds)),
        ("acquisition_channel", *check_acquisition_channel(df)),
    ]
    return results


def report_and_exit_on_failure(results: list[tuple[str, bool, str]]) -> None:
    for name, passed, msg in results:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {msg}")
    if not all(passed for _, passed, _ in results):
        sys.exit(1)
    print("All quality checks passed.")


def assert_all_checks_pass(
    df: pd.DataFrame,
    zero_activity_ids: set,
    expected_count: int,
    signup_dates: dict,
    as_of,
    bounds: dict = SPEC_BOUNDS,
) -> None:
    """Run all checks, print results, SystemExit(1) on any failure."""
    report_and_exit_on_failure(run_all_checks(df, zero_activity_ids, expected_count, signup_dates, as_of, bounds))


def main(argv=None) -> None:
    from features import ingest
    from features.build import DEFAULT_SOURCE_PATHS, build_features

    parser = argparse.ArgumentParser(description="Run feature quality checks standalone.")
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--drift", action="store_true", help="Also check PSI drift vs the feature store's latest versions")
    args = parser.parse_args(argv)
    as_of = pd.Timestamp(args.as_of)

    from feature_store import store

    # Build with the parameters of the store's latest versions — the same definitions a
    # model trained from the store sees — so drift compares like with like.
    df = build_features(as_of, params=store.latest_parameters([vf.name for vf in VERSIONED_FEATURES]))
    customers = ingest.load_customers_csv(DEFAULT_SOURCE_PATHS["customers"])
    customers = customers[customers["signup_date"] < as_of]
    txns = ingest.load_transactions_sqlite(DEFAULT_SOURCE_PATHS["transactions_db"], as_of=as_of)
    results = run_all_checks(
        df,
        zero_activity_ids_from_raw(txns, customers["customer_id"], as_of),
        len(customers),
        dict(zip(customers["customer_id"], customers["signup_date"])),
        as_of,
    )
    if args.drift:
        reference = {f: store.get_latest(f)[f] for f in get_feature_names()}
        results.append(("drift_vs_store_latest", *check_drift(df, reference)))
    report_and_exit_on_failure(results)


if __name__ == "__main__":
    main()
