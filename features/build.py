"""
build.py — Single shared entry point for both training and scoring.

build_features(as_of, source_paths, customer_ids, params) is the one function that both
scripts/build_training_features.py and scripts/score_customer.py call. There is no
"training mode" vs "scoring mode": every feature is always computed for the whole
population with the same code, and filtering by customer_ids at the very end is the only
difference between the two call sites.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from features import ingest, transforms
from features.schema import default_params, get_feature_names


DEFAULT_SOURCE_PATHS = {
    "customers": "data/customers.csv",
    "transactions_db": "data/transactions.db",
    "transactions_parquet": "data/transactions_monthly_agg.parquet",
}


def build_features(
    as_of,
    source_paths: Optional[dict[str, str]] = None,
    customer_ids: Optional[list[int]] = None,
    params: Optional[dict[str, dict]] = None,
) -> pd.DataFrame:
    """Build the full feature set for the given as_of date.

    Parameters
    ----------
    as_of : pd.Timestamp or str
        Required. No transaction on or after this instant is used, and only customers who
        signed up before it are in the population.
    source_paths : dict, optional
        Override data paths. Keys: customers, transactions_db, transactions_parquet.
    customer_ids : list[int], optional
        If provided, only these customers' rows are returned (scoring). None = everyone
        (training). This is the only training/scoring difference.
    params : dict, optional
        Per-feature parameter overrides, e.g. {"txn_count_window": {"window_days": 30}}.
        Unspecified versioned features use their first spec'd version's parameters.

    Returns
    -------
    pd.DataFrame with columns: customer_id, <feature columns in schema order>
    """
    if as_of is None:
        raise ValueError("as_of is required; it is never defaulted inside the feature module.")
    as_of = pd.Timestamp(as_of)

    feature_params = default_params()
    for name, overrides in (params or {}).items():
        if name not in feature_params:
            raise KeyError(f"Unknown versioned feature in params: {name}")
        feature_params[name] = {**feature_params[name], **overrides}

    paths = {**DEFAULT_SOURCE_PATHS, **(source_paths or {})}
    customers = ingest.load_customers_csv(paths["customers"])
    customers = customers[customers["signup_date"] < as_of].reset_index(drop=True)
    transactions = ingest.load_transactions_sqlite(paths["transactions_db"], as_of=as_of)
    # The monthly parquet is deliberately not read by any transform: feature_spec.md's
    # multi-source section rules it out because it is behind the transactions table for
    # July 2026 and missing 10 customers. Its loader stays in ingest.py for reconciliation only.

    t = transforms
    features = pd.DataFrame(index=pd.Index(customers["customer_id"], name="customer_id"))
    features["life_span_days"] = t.compute_life_span_days(customers, transactions, as_of)
    features["return_count_recent"] = t.compute_return_count_recent(
        customers, transactions, as_of, **feature_params["return_count_recent"]
    )
    features["spend_ratio_last_3m_vs_previous"] = t.compute_spend_ratio_last_3m_vs_previous(
        customers, transactions, as_of
    )
    features["txn_ratio_last_3m_vs_previous"] = t.compute_txn_ratio_last_3m_vs_previous(
        customers, transactions, as_of
    )
    features["acquisition_channel"] = t.compute_acquisition_channel(customers, transactions, as_of)
    features["has_txn_history"] = t.compute_has_txn_history(customers, transactions, as_of)
    features["days_since_last_txn"] = t.compute_days_since_last_txn(customers, transactions, as_of)
    features["txn_count_window"] = t.compute_txn_count_window(
        customers, transactions, as_of, **feature_params["txn_count_window"]
    )
    features["return_pct"] = t.compute_return_pct(customers, transactions, as_of)

    features = features[get_feature_names()].reset_index()

    if customer_ids is not None:
        features = features[features["customer_id"].isin(customer_ids)].reset_index(drop=True)

    return features
