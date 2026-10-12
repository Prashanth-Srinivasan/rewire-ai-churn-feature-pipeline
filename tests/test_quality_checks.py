"""
Every check in features/quality_checks.py fires on a deliberately bad input (injected null,
wrong zero-activity default, out-of-range value, unknown channel, duplicated / dropped rows,
shifted distribution) and does not false-positive on the clean real build.
"""

import numpy as np
import pandas as pd
import pytest

from features import ingest
from features import quality_checks as qc
from features.build import build_features
from scripts.build_training_features import TRAINING_AS_OF

AS_OF = pd.Timestamp(TRAINING_AS_OF)


@pytest.fixture(scope="module")
def clean():
    df = build_features(AS_OF)
    customers = ingest.load_customers_csv("data/customers.csv")
    customers = customers[customers["signup_date"] < AS_OF]
    txns = ingest.load_transactions_sqlite("data/transactions.db", as_of=AS_OF)
    zero = qc.zero_activity_ids_from_raw(txns, customers["customer_id"], AS_OF)
    signup = dict(zip(customers["customer_id"], customers["signup_date"]))
    return df, zero, signup


def failed(results):
    return {name for name, passed, _ in results if not passed}


def run(df, zero, signup, expected=None):
    # The population is every customer with signup_date < as_of (feature_spec.md row-count check).
    return qc.run_all_checks(df, zero, len(signup) if expected is None else expected, signup, AS_OF)


def test_checks_do_not_false_positive_on_clean_input(clean):
    df, zero, signup = clean
    assert failed(run(df, zero, signup)) == set()


def test_population_matches_the_figures_stated_in_the_spec(clean):
    """feature_spec.md: 1,400 customers with signup_date < 2026-05-03, 230 with no transaction
    before it. If the data or the training as_of changes, update the spec, then this test."""
    _, zero, signup = clean
    if AS_OF != pd.Timestamp("2026-05-03"):
        pytest.skip("spec figures are stated for the 2026-05-03 training as_of")
    assert (len(signup), len(zero)) == (1400, 230)


def test_null_check_fires_on_injected_null(clean):
    df, zero, signup = clean
    bad = df.copy()
    bad.loc[bad["customer_id"] == next(iter(zero)), "spend_ratio_last_3m_vs_previous"] = np.nan
    ok, msg = qc.check_no_nulls(bad, ["spend_ratio_last_3m_vs_previous"])
    assert not ok and str(next(iter(zero))) in msg
    assert "no_nulls" in failed(run(bad, zero, signup))


def test_null_check_fires_on_missing_column(clean):
    df, zero, signup = clean
    assert "no_nulls" in failed(run(df.drop(columns=["return_pct"]), zero, signup))


def test_zero_activity_check_fires_on_blanket_fillna_zero(clean):
    """The failure the spec names: fillna(0) on the ratios instead of their 1.0 default."""
    df, zero, signup = clean
    bad = df.copy()
    mask = bad["customer_id"].isin(zero)
    bad.loc[mask, "txn_ratio_last_3m_vs_previous"] = 0.0
    assert failed(run(bad, zero, signup)) == {"zero_activity_defaults"}


def test_zero_activity_days_since_check_fires_on_wrong_default(clean):
    df, zero, signup = clean
    bad = df.copy()
    bad.loc[bad["customer_id"].isin(zero), "days_since_last_txn"] = 0
    assert "zero_activity_days_since_last_txn" in failed(run(bad, zero, signup))


@pytest.mark.parametrize("feature, value", [
    ("spend_ratio_last_3m_vs_previous", 501.0),
    ("txn_ratio_last_3m_vs_previous", -0.1),
    ("return_pct", 100.5),
    ("return_count_recent", 11),
    ("txn_count_window", 51),
    ("life_span_days", -3),          # a transaction dated before signup
    ("life_span_days", 901),
    ("days_since_last_txn", -1),     # a window that ignores as_of
])
def test_outlier_check_fires_on_out_of_range_value(clean, feature, value):
    df, zero, signup = clean
    bad = df.copy()
    row = bad.index[bad["has_txn_history"] == 1][0]
    bad[feature] = bad[feature].astype(type(value)) if isinstance(value, float) else bad[feature]
    bad.loc[row, feature] = value
    ok, msg = qc.check_bounds(bad)
    assert not ok and feature in msg


def test_bounds_check_reports_both_sides_of_one_feature(clean):
    df, _, _ = clean
    bad = df.copy()
    bad.loc[0, "return_pct"] = -1.0
    bad.loc[1, "return_pct"] = 101.0
    ok, msg = qc.check_bounds(bad)
    assert not ok and "'rows': 2" in msg


def test_channel_check_fires_on_unknown_or_missing_value(clean):
    df, zero, signup = clean
    for value in ["affiliate", None]:
        bad = df.copy()
        bad["acquisition_channel"] = bad["acquisition_channel"].astype(object)
        bad.loc[5, "acquisition_channel"] = value
        assert "acquisition_channel" in failed(run(bad, zero, signup))


def test_row_count_check_fires_on_duplicated_and_dropped_customers(clean):
    df, zero, signup = clean
    duplicated = pd.concat([df, df.head(1)], ignore_index=True)
    assert "row_count" in failed(run(duplicated, zero, signup))
    # inner join dropping the zero-activity customers
    dropped = df[~df["customer_id"].isin(zero)]
    assert "row_count" in failed(run(dropped, zero, signup))
    # one customer dropped and another duplicated keeps the row count right — still caught
    swapped = pd.concat([df.iloc[2:], df.head(1), df.head(1)], ignore_index=True)
    assert len(swapped) == len(signup)
    assert "row_count" in failed(run(swapped, zero, signup))


def test_drift_check_fires_on_shifted_distribution(clean):
    df, _, _ = clean
    reference = {f: df[f] for f in ["txn_count_window", "spend_ratio_last_3m_vs_previous", "acquisition_channel"]}
    assert qc.check_drift(df, reference)[0]
    shifted = df.copy()
    shifted["txn_count_window"] = shifted["txn_count_window"] + 3
    ok, msg = qc.check_drift(shifted, reference)
    assert not ok and "txn_count_window" in msg
    recat = df.copy()
    recat["acquisition_channel"] = "organic"
    ok, msg = qc.check_drift(recat, reference)
    assert not ok and "acquisition_channel" in msg


def test_drift_check_tolerates_small_noise(clean):
    df, _, _ = clean
    reference = {"spend_ratio_last_3m_vs_previous": df["spend_ratio_last_3m_vs_previous"]}
    resampled = df.sample(frac=1.0, replace=True, random_state=1).reset_index(drop=True)
    assert qc.check_drift(resampled, reference)[0]


def test_assert_all_checks_pass_exits_non_zero(clean):
    df, zero, signup = clean
    bad = df.copy()
    bad.loc[0, "return_pct"] = np.nan
    with pytest.raises(SystemExit) as exc:
        qc.assert_all_checks_pass(bad, zero, len(signup), signup, AS_OF)
    assert exc.value.code == 1


def test_standalone_runner_passes_on_real_data():
    qc.main(["--as-of", TRAINING_AS_OF, "--drift"])  # raises SystemExit(1) on any failure


# ---- added by the coverage-of-intent check (see PARITY_CHECK.md)

def test_row_count_check_matches_requested_ids_on_the_scoring_path(clean):
    """Spec: for scoring, the row count must match the requested customer_ids."""
    _, zero, signup = clean
    requested = [2, 8, 17, 99999]  # 99999 does not exist
    scored = build_features(AS_OF, customer_ids=requested)
    assert not qc.check_row_count(scored, len(requested))[0]
    assert qc.check_row_count(scored, len(requested) - 1)[0]
