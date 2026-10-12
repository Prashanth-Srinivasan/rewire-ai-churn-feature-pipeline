"""
transforms.py — One pure function per feature. No I/O.

Every function takes the customers DataFrame, the transactions DataFrame and as_of,
plus any feature-specific parameters, and returns a Series indexed by customer_id with
one value for every row in `customers`.

as_of is enforced inside every transform (rows on or after as_of are dropped here as well
as at ingestion), so a transform called directly in a test or notebook can never leak.

Zero-activity defaults are applied explicitly — never bare NaN/null. Window conventions
are documented in features/schema.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


RECENT_DAYS = 90       # "last 3 months" in feature_spec.md
DAYS_PER_MONTH = 30    # "for monthly averages, a month is 30 days"


def _before(transactions: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Rows strictly before as_of — the only rows any feature may see."""
    return transactions[transactions["txn_timestamp"] < as_of]


def _last_txn(transactions: pd.DataFrame) -> pd.Series:
    """Each customer's last transaction timestamp (any row, returned or not)."""
    return transactions.groupby("customer_id")["txn_timestamp"].max().rename("last_txn")


def _per_customer(customers: pd.DataFrame, values: pd.Series, default) -> pd.Series:
    """Reindex a per-customer Series onto every customer, filling the zero-activity default."""
    return values.reindex(customers["customer_id"]).fillna(default).rename_axis("customer_id")


def _with_last_txn(customers: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """Transactions joined with each customer's last_txn and signup_date."""
    last = _last_txn(transactions).reset_index()
    return (
        transactions.merge(last, on="customer_id")
        .merge(customers[["customer_id", "signup_date"]], on="customer_id")
    )


def _ratio_recent_vs_previous(
    customers: pd.DataFrame,
    rows: pd.DataFrame,
    all_txns: pd.DataFrame,
    value_col: str | None,
) -> pd.Series:
    """Shared computation for the two *_ratio_last_3m_vs_previous features.

    rows      — the rows whose `value_col` (or row count, if None) is aggregated.
    all_txns  — every row before as_of; defines last_txn so both ratios anchor on the same date.

    recent    = (last_txn - 90d, last_txn]          -> monthly = total / 3
    previous  = [signup_date, last_txn - 90d]       -> monthly = total / (days / 30)
    Default 1.0 when the customer has no transactions, no previous period, or nothing in it.
    """
    last = _last_txn(all_txns)
    cust = customers[["customer_id", "signup_date"]].set_index("customer_id").join(last)

    r = rows.merge(cust.reset_index(), on="customer_id")
    recent_start = r["last_txn"] - pd.Timedelta(days=RECENT_DAYS)
    vals = r[value_col] if value_col else pd.Series(1.0, index=r.index)
    is_recent = r["txn_timestamp"] > recent_start
    is_previous = ~is_recent & (r["txn_timestamp"] >= r["signup_date"])

    recent_total = vals[is_recent].groupby(r.loc[is_recent, "customer_id"]).sum()
    previous_total = vals[is_previous].groupby(r.loc[is_previous, "customer_id"]).sum()

    cust["recent_total"] = recent_total.reindex(cust.index).fillna(0.0)
    cust["previous_total"] = previous_total.reindex(cust.index).fillna(0.0)
    previous_days = (
        (cust["last_txn"] - pd.Timedelta(days=RECENT_DAYS) - cust["signup_date"]).dt.total_seconds() / 86400
    )
    has_previous = cust["last_txn"].notna() & (previous_days > 0) & (cust["previous_total"] > 0)

    recent_monthly = cust["recent_total"] / (RECENT_DAYS / DAYS_PER_MONTH)
    previous_monthly = cust["previous_total"] / (previous_days / DAYS_PER_MONTH)
    ratio = pd.Series(1.0, index=cust.index)
    ratio[has_previous] = recent_monthly[has_previous] / previous_monthly[has_previous]
    return ratio.astype("float64").rename_axis("customer_id")


def compute_life_span_days(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.Series:
    """Days from signup_date to the customer's last transaction before as_of.

    Zero-activity default: 0.
    """
    last = _last_txn(_before(transactions, as_of))
    signup = customers.set_index("customer_id")["signup_date"]
    span = (last - signup.reindex(last.index)).dt.days
    return _per_customer(customers, span, 0).astype("int64")


def compute_return_count_recent(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    as_of: pd.Timestamp,
    window_days: int,
) -> pd.Series:
    """Returned rows in (last_txn - window_days, last_txn], last_txn = last transaction before as_of.

    Zero-activity default: 0.
    """
    txns = _with_last_txn(customers, _before(transactions, as_of))
    in_window = txns[
        (txns["is_returned"] == 1)
        & (txns["txn_timestamp"] > txns["last_txn"] - pd.Timedelta(days=window_days))
    ]
    counts = in_window.groupby("customer_id").size()
    return _per_customer(customers, counts, 0).astype("int64")


def compute_spend_ratio_last_3m_vs_previous(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.Series:
    """Average monthly spend (is_returned = 0) last 3 months / previous period.

    The 90-day window ends on the customer's last transaction of any kind (per spec), so a
    customer whose most recent rows are returns can get a ratio of 0, not the 1.0 default.
    Zero-activity default: 1.0 (also when no previous period, or no spend in it).
    """
    txns = _before(transactions, as_of)
    purchases = txns[txns["is_returned"] == 0]
    return _ratio_recent_vs_previous(customers, purchases, txns, value_col="amount")


def compute_txn_ratio_last_3m_vs_previous(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.Series:
    """Average monthly transaction count (all rows) last 3 months / previous period.

    Zero-activity default: 1.0 (also when no previous period, or no transaction in it).
    """
    txns = _before(transactions, as_of)
    return _ratio_recent_vs_previous(customers, txns, txns, value_col=None)


def compute_acquisition_channel(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.Series:
    """Acquisition channel as a text category.

    No default: a static attribute. Missing/unknown values are left as-is so the
    quality check fails on them rather than being masked here.
    """
    return customers.set_index("customer_id")["acquisition_channel"]


def compute_has_txn_history(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.Series:
    """1 if the customer has at least one transaction before as_of, else 0."""
    ids = _before(transactions, as_of)["customer_id"].unique()
    flag = customers["customer_id"].isin(ids).astype("int64")
    return pd.Series(flag.values, index=customers["customer_id"], name="has_txn_history")


def compute_days_since_last_txn(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.Series:
    """Days from the customer's last transaction before as_of to as_of.

    Zero-activity default: days from signup_date to as_of.
    """
    last = _last_txn(_before(transactions, as_of)).reindex(customers["customer_id"])
    signup = customers.set_index("customer_id")["signup_date"]
    anchor = last.fillna(signup)
    return (as_of - anchor).dt.days.astype("int64").rename_axis("customer_id")


def compute_txn_count_window(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    as_of: pd.Timestamp,
    window_days: int,
) -> pd.Series:
    """Rows (returns included) with txn_timestamp in [as_of - window_days, as_of).

    Zero-activity default: 0.
    """
    txns = _before(transactions, as_of)
    in_window = txns[txns["txn_timestamp"] >= as_of - pd.Timedelta(days=window_days)]
    counts = in_window.groupby("customer_id").size()
    return _per_customer(customers, counts, 0).astype("int64")


def compute_return_pct(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.Series:
    """100 * returned rows / all rows before as_of.

    Zero-activity default: 0.0.
    """
    pct = _before(transactions, as_of).groupby("customer_id")["is_returned"].mean() * 100
    return _per_customer(customers, pct, 0.0).astype("float64")
