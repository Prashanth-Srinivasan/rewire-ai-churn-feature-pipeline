"""
Shared fixtures: a small hand-built dataset whose expected feature values are worked out by
hand in test_transforms.py, plus helpers to write it to real CSV / SQLite / parquet files so
build_features() can be driven through its normal ingestion path.
"""

import sqlite3

import pandas as pd
import pytest

AS_OF = pd.Timestamp("2026-08-01")


def _t(s):
    return pd.Timestamp(s)


@pytest.fixture
def customers():
    return pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5, 6, 7],
        "signup_date": pd.to_datetime([
            "2025-01-01",  # 1: normal history
            "2026-03-01",  # 2: zero activity
            "2026-01-01",  # 3: single purchase
            "2026-01-01",  # 4: single row, a return
            "2026-02-01",  # 5: only transaction is after as_of -> zero activity at as_of
            "2025-01-01",  # 6: last rows are returns, no purchase in the recent window
            "2026-06-01",  # 7: tenure < 90 days at last txn -> no previous period
        ]),
        "acquisition_channel": ["organic", "paid_search", "referral", "social", "organic", "social", "referral"],
        "region": ["east"] * 7,
        "churned": [0, 1, 0, 0, 1, 1, 0],
    })


@pytest.fixture
def transactions():
    rows = [
        # customer 1 (worked example in test_transforms.py)
        (1, "2025-03-01 00:00:00", 100.0, 0),
        (1, "2025-06-01 00:00:00", 50.0, 1),
        (1, "2026-05-15 00:00:00", 60.0, 0),
        (1, "2026-07-01 00:00:00", 30.0, 0),
        (1, "2026-07-11 00:00:00", 20.0, 1),
        (1, "2026-08-05 00:00:00", 999.0, 0),   # after as_of: must never be seen
        # customer 3
        (3, "2026-07-01 00:00:00", 50.0, 0),
        # customer 4
        (4, "2026-07-01 00:00:00", 40.0, 1),
        # customer 5
        (5, "2026-08-02 00:00:00", 70.0, 0),
        # customer 6: purchases long ago, recent rows are returns only
        (6, "2025-03-01 00:00:00", 200.0, 0),
        (6, "2026-07-01 00:00:00", 10.0, 1),
        (6, "2026-07-20 00:00:00", 15.0, 1),
        # customer 7
        (7, "2026-07-01 00:00:00", 25.0, 0),
        (7, "2026-07-15 00:00:00", 35.0, 0),
    ]
    df = pd.DataFrame(rows, columns=["customer_id", "txn_timestamp", "amount", "is_returned"])
    df["txn_timestamp"] = pd.to_datetime(df["txn_timestamp"])
    df["category"] = "home"
    df["channel"] = "web"
    return df[["customer_id", "txn_timestamp", "category", "channel", "amount", "is_returned"]]


def write_sources(tmp_path, customers, transactions):
    """Write the frames in the raw formats the real sources use; return source_paths."""
    paths = {
        "customers": str(tmp_path / "customers.csv"),
        "transactions_db": str(tmp_path / "transactions.db"),
        "transactions_parquet": str(tmp_path / "monthly.parquet"),
    }
    c = customers.copy()
    c["signup_date"] = c["signup_date"].dt.strftime("%Y-%m-%d")
    c.to_csv(paths["customers"], index=False)
    t = transactions.copy()
    # Same text format as data/transactions.db (space separator, fractional seconds).
    t["txn_timestamp"] = t["txn_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    with sqlite3.connect(paths["transactions_db"]) as conn:
        t.to_sql("transactions", conn, index=False)
    pd.DataFrame({"customer_id": [1], "txn_month": [pd.Timestamp("2026-07-01")], "txn_count": [1],
                  "total_amount": [1.0], "return_count": [0]}).to_parquet(paths["transactions_parquet"])
    return paths


@pytest.fixture
def synthetic_sources(tmp_path, customers, transactions):
    return write_sources(tmp_path, customers, transactions)
