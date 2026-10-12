"""
ingest.py — One loader function per source (CSV / SQLite / parquet).

Pure I/O: no feature logic. Each loader returns a pandas DataFrame.
Sources are never modified.
"""

from __future__ import annotations

import sqlite3
from typing import Optional

import pandas as pd


def load_customers_csv(path: str = "data/customers.csv") -> pd.DataFrame:
    """Load customer profile data from CSV.

    Returns columns: customer_id, signup_date, acquisition_channel, region, churned.
    """
    df = pd.read_csv(path)
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    return df


def load_transactions_sqlite(
    db_path: str = "data/transactions.db",
    table: str = "transactions",
    as_of: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Load transaction rows from SQLite.

    If as_of is provided, only transactions strictly before as_of are returned.
    Returns columns: customer_id, txn_timestamp, category, channel, amount, is_returned.

    The as_of filter is applied to the *parsed* timestamps, not as a SQL string comparison:
    the table stores '2026-05-01 00:00:05' (space separator), which sorts before an ISO
    '2026-05-01T00:00:00' string, so a string filter would leak the whole as_of day.
    """
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
    finally:
        conn.close()
    df["txn_timestamp"] = pd.to_datetime(df["txn_timestamp"])
    if as_of is not None:
        df = df[df["txn_timestamp"] < pd.Timestamp(as_of)].reset_index(drop=True)
    return df


def load_monthly_agg_parquet(path: str = "data/transactions_monthly_agg.parquet") -> pd.DataFrame:
    """Load pre-aggregated monthly transaction data from parquet.

    Returns columns: customer_id, txn_month, txn_count, total_amount, return_count.
    """
    return pd.read_parquet(path)