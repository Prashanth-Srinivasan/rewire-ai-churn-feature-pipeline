"""
Each feature function in features/transforms.py in isolation, against the hand-built
fixture in conftest.py: normal case, zero-activity case, single-record case, and
time-window boundaries. Expected values are worked out by hand in the comments.

Customer 1 (signup 2025-01-01), rows before as_of = 2026-08-01:
    2025-03-01 buy 100 | 2025-06-01 return 50 | 2026-05-15 buy 60 | 2026-07-01 buy 30 | 2026-07-11 return 20
    (+ 2026-08-05 buy 999, after as_of, must be ignored)
    last_txn = 2026-07-11; recent = (2026-04-12, 2026-07-11]; previous = [2025-01-01, 2026-04-12] = 466 days
"""

import sqlite3

import numpy as np
import pandas as pd
import pytest

from features import ingest
from features import transforms as t
from tests.conftest import AS_OF


def at(series, cid):
    return series.loc[cid]


# ---------------------------------------------------------------- normal case (customer 1)

def test_normal_case_all_features(customers, transactions):
    prev_months = 466 / 30
    assert at(t.compute_life_span_days(customers, transactions, AS_OF), 1) == 556
    assert at(t.compute_days_since_last_txn(customers, transactions, AS_OF), 1) == 21
    assert at(t.compute_has_txn_history(customers, transactions, AS_OF), 1) == 1
    assert at(t.compute_return_pct(customers, transactions, AS_OF), 1) == pytest.approx(40.0)
    assert at(t.compute_return_count_recent(customers, transactions, AS_OF, window_days=90), 1) == 1
    assert at(t.compute_return_count_recent(customers, transactions, AS_OF, window_days=500), 1) == 2
    assert at(t.compute_txn_count_window(customers, transactions, AS_OF, window_days=30), 1) == 1
    assert at(t.compute_txn_count_window(customers, transactions, AS_OF, window_days=90), 1) == 3
    # spend: recent purchases 60 + 30 = 90 over 3 months; previous purchases 100 over 466/30 months
    assert at(t.compute_spend_ratio_last_3m_vs_previous(customers, transactions, AS_OF), 1) == pytest.approx(
        (90 / 3) / (100 / prev_months))
    # count: 3 recent rows over 3 months; 2 previous rows (returns included) over 466/30 months
    assert at(t.compute_txn_ratio_last_3m_vs_previous(customers, transactions, AS_OF), 1) == pytest.approx(
        (3 / 3) / (2 / prev_months))
    assert at(t.compute_acquisition_channel(customers, transactions, AS_OF), 1) == "organic"


def test_rows_on_or_after_as_of_are_ignored(customers, transactions):
    """The 999 purchase on 2026-08-05 must not move any feature; moving as_of past it must."""
    later = pd.Timestamp("2026-08-10")
    assert at(t.compute_days_since_last_txn(customers, transactions, AS_OF), 1) == 21
    assert at(t.compute_days_since_last_txn(customers, transactions, later), 1) == 5
    assert at(t.compute_txn_count_window(customers, transactions, AS_OF, window_days=90), 1) == 3


# ---------------------------------------------------------------- zero activity (customers 2 and 5)

@pytest.mark.parametrize("cid, days_since_signup", [(2, 153), (5, 181)])
def test_zero_activity_defaults(customers, transactions, cid, days_since_signup):
    """Customer 5's only row is after as_of, so it is zero-activity at as_of too."""
    expected = {
        t.compute_life_span_days: 0,
        t.compute_has_txn_history: 0,
        t.compute_return_pct: 0.0,
        t.compute_spend_ratio_last_3m_vs_previous: 1.0,
        t.compute_txn_ratio_last_3m_vs_previous: 1.0,
    }
    for fn, value in expected.items():
        out = fn(customers, transactions, AS_OF)
        assert not out.isnull().any(), fn.__name__
        assert at(out, cid) == value, fn.__name__
    assert at(t.compute_return_count_recent(customers, transactions, AS_OF, window_days=180), cid) == 0
    assert at(t.compute_txn_count_window(customers, transactions, AS_OF, window_days=30), cid) == 0
    assert at(t.compute_days_since_last_txn(customers, transactions, AS_OF), cid) == days_since_signup


def test_every_customer_gets_a_row_even_with_no_transactions_at_all(customers):
    empty = pd.DataFrame({"customer_id": pd.Series(dtype="int64"),
                          "txn_timestamp": pd.Series(dtype="datetime64[ns]"),
                          "amount": pd.Series(dtype="float64"), "is_returned": pd.Series(dtype="int64")})
    for fn in [t.compute_life_span_days, t.compute_has_txn_history, t.compute_return_pct,
               t.compute_spend_ratio_last_3m_vs_previous, t.compute_txn_ratio_last_3m_vs_previous,
               t.compute_days_since_last_txn]:
        out = fn(customers, empty, AS_OF)
        assert len(out) == len(customers) and not out.isnull().any(), fn.__name__


# ---------------------------------------------------------------- single record (customers 3 and 4)

def test_single_purchase(customers, transactions):
    """Customer 3: one purchase 2026-07-01, signup 2026-01-01."""
    assert at(t.compute_life_span_days(customers, transactions, AS_OF), 3) == 181
    assert at(t.compute_days_since_last_txn(customers, transactions, AS_OF), 3) == 31
    assert at(t.compute_txn_count_window(customers, transactions, AS_OF, window_days=30), 3) == 0
    assert at(t.compute_txn_count_window(customers, transactions, AS_OF, window_days=90), 3) == 1
    assert at(t.compute_return_pct(customers, transactions, AS_OF), 3) == 0.0
    # a previous period exists (91 days) but holds nothing -> spec default 1.0
    assert at(t.compute_spend_ratio_last_3m_vs_previous(customers, transactions, AS_OF), 3) == 1.0
    assert at(t.compute_txn_ratio_last_3m_vs_previous(customers, transactions, AS_OF), 3) == 1.0


def test_single_return(customers, transactions):
    """Customer 4: one row, a return. It is a transaction (has history) but not spend."""
    assert at(t.compute_has_txn_history(customers, transactions, AS_OF), 4) == 1
    assert at(t.compute_return_pct(customers, transactions, AS_OF), 4) == 100.0
    assert at(t.compute_return_count_recent(customers, transactions, AS_OF, window_days=90), 4) == 1
    assert at(t.compute_spend_ratio_last_3m_vs_previous(customers, transactions, AS_OF), 4) == 1.0


# ---------------------------------------------------------------- ratio edge cases

def test_spend_ratio_window_ends_on_last_transaction_of_any_kind(customers, transactions):
    """Customer 6: last rows are returns. The 90-day window ends on 2026-07-20 (a return), holds
    no purchase, and the previous period holds a 200 purchase -> ratio 0, not the 1.0 default.
    (Regression test for the v1 bug that anchored on the last non-returned purchase.)"""
    assert at(t.compute_spend_ratio_last_3m_vs_previous(customers, transactions, AS_OF), 6) == 0.0
    # count ratio: 2 recent rows / 3 months vs 1 previous row over 475/30 months
    assert at(t.compute_txn_ratio_last_3m_vs_previous(customers, transactions, AS_OF), 6) == pytest.approx(
        (2 / 3) / (1 / (475 / 30)))


def test_ratios_default_when_no_previous_period(customers, transactions):
    """Customer 7: signup 2026-06-01, last txn 2026-07-15 -> under 90 days of tenure, no previous period."""
    assert at(t.compute_spend_ratio_last_3m_vs_previous(customers, transactions, AS_OF), 7) == 1.0
    assert at(t.compute_txn_ratio_last_3m_vs_previous(customers, transactions, AS_OF), 7) == 1.0


# ---------------------------------------------------------------- window boundaries

def _one_customer(signup="2025-01-01"):
    return pd.DataFrame({"customer_id": [1], "signup_date": [pd.Timestamp(signup)],
                         "acquisition_channel": ["organic"]})


def _txns(*rows):
    df = pd.DataFrame(rows, columns=["txn_timestamp", "amount", "is_returned"])
    df["txn_timestamp"] = pd.to_datetime(df["txn_timestamp"])
    df.insert(0, "customer_id", 1)
    return df


def test_txn_count_window_boundaries():
    """[as_of - 30d, as_of): start inclusive, as_of exclusive."""
    c = _one_customer()
    start = AS_OF - pd.Timedelta(days=30)
    assert at(t.compute_txn_count_window(c, _txns((start, 1, 0)), AS_OF, window_days=30), 1) == 1
    assert at(t.compute_txn_count_window(c, _txns((start - pd.Timedelta(microseconds=1), 1, 0)), AS_OF, window_days=30), 1) == 0
    assert at(t.compute_txn_count_window(c, _txns((AS_OF, 1, 0)), AS_OF, window_days=30), 1) == 0


def test_return_count_recent_boundaries():
    """(last_txn - window, last_txn]: a return exactly window days before last_txn is out."""
    c = _one_customer()
    last = pd.Timestamp("2026-07-01")
    edge = last - pd.Timedelta(days=90)
    rows = [(last, 1, 0), (edge, 1, 1)]
    assert at(t.compute_return_count_recent(c, _txns(*rows), AS_OF, window_days=90), 1) == 0
    rows = [(last, 1, 0), (edge + pd.Timedelta(seconds=1), 1, 1)]
    assert at(t.compute_return_count_recent(c, _txns(*rows), AS_OF, window_days=90), 1) == 1


def test_ratio_recent_window_boundary():
    """A row exactly 90 days before last_txn belongs to the previous period, not the recent one."""
    c = _one_customer("2025-01-01")
    last = pd.Timestamp("2026-07-01")
    edge = last - pd.Timedelta(days=90)
    out = t.compute_txn_ratio_last_3m_vs_previous(c, _txns((edge, 1, 0), (last, 1, 0)), AS_OF)
    previous_days = (edge - pd.Timestamp("2025-01-01")).days
    assert at(out, 1) == pytest.approx((1 / 3) / (1 / (previous_days / 30)))


def test_days_since_last_txn_floors_partial_days():
    c = _one_customer()
    out = t.compute_days_since_last_txn(c, _txns((AS_OF - pd.Timedelta(hours=23), 1, 0)), AS_OF)
    assert at(out, 1) == 0


# ---------------------------------------------------------------- ingestion

def test_sqlite_loader_excludes_the_as_of_day(tmp_path):
    """Regression: the DB stores '2026-05-01 00:00:05'; a string filter against an ISO
    '2026-05-01T00:00:00' would have let this row through."""
    db = tmp_path / "t.db"
    with sqlite3.connect(db) as conn:
        pd.DataFrame({"customer_id": [1, 1], "txn_timestamp": ["2026-04-30 23:59:59.9", "2026-05-01 00:00:05.1"],
                      "category": ["home"] * 2, "channel": ["web"] * 2, "amount": [1.0, 2.0],
                      "is_returned": [0, 0]}).to_sql("transactions", conn, index=False)
    out = ingest.load_transactions_sqlite(str(db), as_of=pd.Timestamp("2026-05-01"))
    assert out["amount"].tolist() == [1.0]


def test_loaders_read_real_sources():
    assert len(ingest.load_customers_csv("data/customers.csv")) == 1500
    assert len(ingest.load_transactions_sqlite("data/transactions.db")) == 11742
    assert len(ingest.load_monthly_agg_parquet("data/transactions_monthly_agg.parquet")) == 8086
