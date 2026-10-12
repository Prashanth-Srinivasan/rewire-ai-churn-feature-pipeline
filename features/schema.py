"""
schema.py — Versioned output feature schema.

Each FeatureSpec entry describes one feature in the output DataFrame:
name, dtype, nullable (False for all features because defaults are applied in transforms),
source(s), the feature-set version, and the spec-derived lineage text (computation,
zero-activity default, rationale) that FEATURE_LINEAGE.md and the feature-store registry
are generated from.

Window conventions (shared by every transform, stated once here):
- "Recent" window for the ratio features and return_count_recent: (last_txn - N days, last_txn],
  where last_txn is the customer's last transaction (any row, returned or not) before as_of.
- txn_count_window: [as_of - N days, as_of).
- Nothing on or after as_of is ever read (filtered at ingestion).
"""

from __future__ import annotations

from dataclasses import dataclass, field


FEATURE_SET_VERSION = 1


@dataclass
class FeatureSpec:
    name: str
    dtype: str
    nullable: bool
    sources: list[str]
    description: str
    computation: str
    zero_activity_default: str
    rationale: str
    feature_set_version: int = FEATURE_SET_VERSION


@dataclass
class VersionSpec:
    """One spec-defined version of a versioned feature."""
    parameters: dict
    reason: str


@dataclass
class VersionedFeature:
    name: str
    parameter: str
    versions: list[VersionSpec] = field(default_factory=list)


_RECENT = "90 days ending on the customer's last transaction before as_of, i.e. (last_txn - 90d, last_txn]"

FEATURES: list[FeatureSpec] = [
    FeatureSpec(
        name="life_span_days",
        dtype="int64",
        nullable=False,
        sources=["data/transactions.db:transactions.txn_timestamp", "data/customers.csv:signup_date"],
        description="Days from signup to the customer's last transaction before as_of.",
        computation="floor((max(txn_timestamp where txn_timestamp < as_of) - signup_date) in days)",
        zero_activity_default="0",
        rationale=(
            "Customers who stopped buying soon after signup tend to be one-time buyers who do not "
            "come back; measures how long the customer stayed active, not how much they bought."
        ),
    ),
    FeatureSpec(
        name="return_count_recent",
        dtype="int64",
        nullable=False,
        sources=["data/transactions.db:transactions.is_returned,txn_timestamp"],
        description="Returned rows in the last window_days ending on the customer's last transaction (versioned).",
        computation=(
            "count of rows with is_returned = 1 and txn_timestamp in (last_txn - window_days, last_txn], "
            "last_txn = customer's last transaction (any row) before as_of"
        ),
        zero_activity_default="0",
        rationale=(
            "Customers who returned items in their most recent months may leave after a bad experience; "
            "a customer can be buying normally and still be unhappy with what they receive. A hypothesis the "
            "data does not yet support: at the training as_of, customers with any return churned at 9.8% "
            "against 10.6% without. Kept so a model can test it, and because returns can also mean an "
            "engaged customer."
        ),
    ),
    FeatureSpec(
        name="spend_ratio_last_3m_vs_previous",
        dtype="float64",
        nullable=False,
        sources=[
            "data/transactions.db:transactions.amount,is_returned,txn_timestamp",
            "data/customers.csv:signup_date",
        ],
        description="Average monthly spend in the last 3 months / average monthly spend in the previous period.",
        computation=(
            f"spend = sum(amount where is_returned = 0). recent = {_RECENT}; recent_monthly = recent spend / 3. "
            "previous = [signup_date, last_txn - 90d]; previous_monthly = previous spend / (previous days / 30). "
            "ratio = recent_monthly / previous_monthly"
        ),
        zero_activity_default="1.0 (also when there is no previous period, i.e. last_txn - signup_date <= 90 days, or no spend in it)",
        rationale=(
            "Customers whose recent spend has fallen below their own earlier average tend to be reducing "
            "their business before leaving; a customer who always spent little gets a usual value."
        ),
    ),
    FeatureSpec(
        name="txn_ratio_last_3m_vs_previous",
        dtype="float64",
        nullable=False,
        sources=["data/transactions.db:transactions.txn_timestamp", "data/customers.csv:signup_date"],
        description="Average monthly transaction count in the last 3 months / same in the previous period.",
        computation=(
            f"all rows (returns included). recent = {_RECENT}; recent_monthly = recent count / 3. "
            "previous = [signup_date, last_txn - 90d]; previous_monthly = previous count / (previous days / 30). "
            "ratio = recent_monthly / previous_monthly"
        ),
        zero_activity_default="1.0 (also when there is no previous period or no transaction in it)",
        rationale=(
            "Customers ordering less often than their own earlier pace tend to be losing interest; "
            "a customer who always ordered rarely gets a usual value."
        ),
    ),
    FeatureSpec(
        name="acquisition_channel",
        dtype="object",
        nullable=False,
        sources=["data/customers.csv:acquisition_channel"],
        description="Customer acquisition channel (organic / paid_search / referral / social).",
        computation="copied as a text category; any value outside the four known channels fails the quality check",
        zero_activity_default="not applicable (static attribute)",
        rationale=(
            "Paid-search customers were already looking for a product and tend to stay "
            "(10.2% churn vs 15.3% organic); fixed at signup, independent of activity."
        ),
    ),
    FeatureSpec(
        name="has_txn_history",
        dtype="int64",
        nullable=False,
        sources=["data/transactions.db:transactions.customer_id,txn_timestamp"],
        description="1 if the customer has at least one transaction before as_of, else 0.",
        computation="1 if any row with txn_timestamp < as_of exists for the customer, else 0",
        zero_activity_default="0",
        rationale=(
            "Customers who signed up and never bought tend to drift away without any event; "
            "separates 'never started' from 'started and slowed down'."
        ),
    ),
    FeatureSpec(
        name="days_since_last_txn",
        dtype="int64",
        nullable=False,
        sources=["data/transactions.db:transactions.txn_timestamp", "data/customers.csv:signup_date"],
        description="Days from the customer's last transaction before as_of to as_of.",
        computation="floor((as_of - last_txn) in days)",
        zero_activity_default="floor((as_of - signup_date) in days)",
        rationale=(
            "Customers who have not bought for a long time tend to have moved elsewhere; "
            "a customer with many past orders can still have a long gap now."
        ),
    ),
    FeatureSpec(
        name="txn_count_window",
        dtype="int64",
        nullable=False,
        sources=["data/transactions.db:transactions.txn_timestamp"],
        description="Transaction count in the last window_days before as_of (versioned).",
        computation="count of rows (returns included) with txn_timestamp in [as_of - window_days, as_of)",
        zero_activity_default="0",
        rationale=(
            "Customers ordering less often right now tend to be disengaging; counts visits not money, "
            "and shows the current level where txn_ratio shows change against the customer's own past."
        ),
    ),
    FeatureSpec(
        name="return_pct",
        dtype="float64",
        nullable=False,
        sources=["data/transactions.db:transactions.is_returned,txn_timestamp"],
        description="Percentage of the customer's rows before as_of that are returns.",
        computation="100 * count(is_returned = 1) / count(all rows), over rows with txn_timestamp < as_of",
        zero_activity_default="0.0",
        rationale=(
            "Customers who return a large share of what they buy may stop ordering; as a share, a "
            "regular buyer with many returns is flagged and an occasional buyer with none is not. As with "
            "return_count_recent, the data shows almost no difference in churn for customers with returns, "
            "so this is kept as a hypothesis for the model to test."
        ),
    ),
]


# The versioning exercise exactly as stated in feature_spec.md ("Feature store, versioning & lineage").
# Order matters: versions are registered in list order, so the last entry ends up is_latest.
VERSIONED_FEATURES: list[VersionedFeature] = [
    VersionedFeature(
        name="txn_count_window",
        parameter="window_days",
        versions=[
            VersionSpec({"window_days": 90}, "Initial window: 90 days before as_of."),
            VersionSpec(
                {"window_days": 30},
                "Window shortened from 90 to 30 days: the usual customer buys about twice in 90 days and "
                "last bought 24 days ago, so a 90-day count still looks normal for someone silent for two "
                "months; 30 days shows a recent slowdown sooner.",
            ),
        ],
    ),
    VersionedFeature(
        name="return_count_recent",
        parameter="window_days",
        versions=[
            VersionSpec({"window_days": 90}, "Initial window: 90 days ending on the customer's last transaction."),
            VersionSpec(
                {"window_days": 180},
                "Window extended from 90 to 180 days: returns are rare, so in a 90-day window the count is 0 "
                "for almost every customer (99th percentile is 1); 180 days picks up more returns and "
                "separates customers better.",
            ),
        ],
    ),
]


def default_params() -> dict[str, dict]:
    """Parameters for the *first* spec'd version of every versioned feature."""
    return {vf.name: dict(vf.versions[0].parameters) for vf in VERSIONED_FEATURES}


def get_feature_names(version: int = FEATURE_SET_VERSION) -> list[str]:
    """Return feature names for a given feature-set version."""
    return [f.name for f in FEATURES if f.feature_set_version == version]


def get_feature_spec(name: str) -> FeatureSpec:
    """Get a single feature's spec by name."""
    for f in FEATURES:
        if f.name == name:
            return f
    raise KeyError(f"Unknown feature: {name}")


def get_versioned_feature(name: str) -> VersionedFeature:
    for vf in VERSIONED_FEATURES:
        if vf.name == name:
            return vf
    raise KeyError(f"{name} is not a versioned feature")
