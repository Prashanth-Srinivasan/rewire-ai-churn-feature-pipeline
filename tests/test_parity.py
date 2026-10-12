"""
Train/score parity: build features for the same customer via the "training" path (one
batch over the whole population) and the "scoring" path (one customer_id at a time), same
as_of, and assert every feature value is identical — exact equality, not approximate.

Run against both the real data (historical and training as_of dates) and the synthetic
fixture, whose timestamps sit exactly on window boundaries: a scoring-only change to an
inclusive/exclusive boundary would never show up on the real data's microsecond timestamps.

If these paths ever disagree, fix features/build.py — not this test.
"""

import json

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from feature_store import store
from features import ingest
from features.build import build_features
from features.schema import VERSIONED_FEATURES, get_feature_names
from scripts.build_training_features import TRAINING_AS_OF
from tests.conftest import write_sources

# The training as_of, and the day after the latest transaction (what scoring "today" sees).
REAL_AS_OFS = [TRAINING_AS_OF, "2026-08-01"]


def _sample_ids(training: pd.DataFrame, as_of) -> list[int]:
    """A spread of customers: anyone with raw rows on/just after as_of (where a scoring-only
    leak would show), zero-activity, short-tenure, heavy, returners, and a fixed sample."""
    raw = ingest.load_transactions_sqlite("data/transactions.db")
    as_of = pd.Timestamp(as_of)
    near = raw[(raw["txn_timestamp"] >= as_of) & (raw["txn_timestamp"] < as_of + pd.Timedelta(days=2))]
    picks = set(near["customer_id"]) & set(training["customer_id"])
    picks |= set(training.loc[training["has_txn_history"] == 0, "customer_id"].head(5))
    picks |= set(training.nsmallest(5, "life_span_days")["customer_id"])
    picks |= set(training.nlargest(5, "txn_count_window")["customer_id"])
    picks |= set(training.nlargest(5, "return_pct")["customer_id"])
    picks |= set(training.nsmallest(5, "spend_ratio_last_3m_vs_previous")["customer_id"])
    picks |= set(training["customer_id"].sample(25, random_state=0))
    return sorted(int(c) for c in picks)


def _assert_parity(training: pd.DataFrame, ids, **build_kwargs):
    # 1) the scoring path for the whole population in one call (catches anything scoring-only)
    everyone = build_features(customer_ids=training["customer_id"].tolist(), **build_kwargs)
    assert_frame_equal(everyone, training, check_exact=True)
    # 2) the scoring path one customer at a time, as score_customer.py is used
    scored = pd.concat(
        [build_features(customer_ids=[cid], **build_kwargs) for cid in ids], ignore_index=True
    )
    assert len(scored) == len(ids), "scoring path dropped a customer the training path kept"
    expected = training[training["customer_id"].isin(ids)].sort_values("customer_id").reset_index(drop=True)
    assert list(scored.columns) == ["customer_id", *get_feature_names()]
    assert_frame_equal(scored.sort_values("customer_id").reset_index(drop=True), expected, check_exact=True)


@pytest.mark.parametrize("as_of", REAL_AS_OFS)
@pytest.mark.parametrize("version_index", [0, -1])
def test_training_and_scoring_paths_agree_on_real_data(as_of, version_index):
    params = {vf.name: vf.versions[version_index].parameters for vf in VERSIONED_FEATURES}
    training = build_features(as_of=as_of, params=params)
    _assert_parity(training, _sample_ids(training, as_of), as_of=as_of, params=params)


def test_training_and_scoring_paths_agree_on_boundary_fixture(customers, transactions, tmp_path):
    # Add rows that sit exactly on every window edge for customer 1 at as_of 2026-08-01.
    as_of = pd.Timestamp("2026-08-01")
    last = pd.Timestamp("2026-07-11")
    edges = [as_of - pd.Timedelta(days=30), as_of - pd.Timedelta(days=90),
             last - pd.Timedelta(days=90), last - pd.Timedelta(days=180), as_of]
    extra = pd.DataFrame({"customer_id": 1, "txn_timestamp": edges, "category": "home", "channel": "web",
                          "amount": 5.0, "is_returned": [1, 0, 1, 1, 0]})
    paths = write_sources(tmp_path, customers, pd.concat([transactions, extra], ignore_index=True))
    for i in (0, -1):
        params = {vf.name: vf.versions[i].parameters for vf in VERSIONED_FEATURES}
        training = build_features(as_of=as_of, source_paths=paths, params=params)
        _assert_parity(training, training["customer_id"].tolist(), as_of=as_of, source_paths=paths, params=params)


def test_scoring_with_store_params_reproduces_the_registered_latest_snapshots():
    """End to end: what scripts/score_customer.py computes for a customer today equals what
    scripts/build_training_features.py registered as each feature's latest version."""
    from scripts.score_customer import latest_store_params

    params = latest_store_params()
    assert params == {vf.name: vf.versions[-1].parameters for vf in VERSIONED_FEATURES}
    ids = [1, 2, 8, 17, 697, 1500]
    for feature in get_feature_names():
        snapshot = store.get_latest(feature)
        (as_of,) = snapshot["as_of_date"].unique()  # score at the date the snapshot was built
        scored = build_features(as_of=as_of, customer_ids=ids, params=params).set_index("customer_id")
        stored = snapshot.set_index("customer_id").loc[ids, feature]
        assert str(stored.dtype) == str(scored[feature].dtype), feature
        assert (stored.values == scored[feature].values).all(), feature


def test_as_of_is_required():
    with pytest.raises(ValueError):
        build_features(as_of=None)


# ---- added by the coverage-of-intent check (see PARITY_CHECK.md)

def test_no_feature_reads_the_monthly_parquet(synthetic_sources):
    """Spec multi-source section: every feature comes from transactions.db, never the parquet.
    Pointing the parquet path at a file that does not exist must not change the build."""
    as_of = "2026-08-01"
    expected = build_features(as_of, source_paths=synthetic_sources)
    broken = {**synthetic_sources, "transactions_parquet": "does/not/exist.parquet"}
    assert_frame_equal(build_features(as_of, source_paths=broken), expected, check_exact=True)
