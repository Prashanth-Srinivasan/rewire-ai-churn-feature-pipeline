# Candidate Feature Proposal — Customer Churn Prediction

## Objective

Propose a set of candidate features for predicting customer churn in an e-commerce setting. The proposal is informed by the raw data sources (`customers.csv`, `transactions.db`, `transactions_monthly_agg.parquet`) documented in `data_dictionary.md`.

The dataset has ~13.5% churn rate across 1,500 customers with up to 2.5 years of transaction history. ~81 customers have zero recorded transactions, which the feature design must handle explicitly.

---

## Design Principles

1. **Shared features** — One feature set serves both training (bulk) and scoring (single customer) paths. No "training mode" vs "scoring mode" special-casing.
2. **As-of semantics** — Features are computed relative to an `as_of` date. For training, this would be each customer's churn determination date; for scoring, it's the current time.
3. **Zero-activity defaults** — Customers with no transaction history (or activity segments that are empty) get explicit sentinel values, never NaN/null.
4. **Versioned schema** — All features have a version so lineage is tractable through the feature store.

---

## Proposed Features

### Group A: Static / Profile Features (from `customers.csv`)

These are known at signup and do not change over time.

| # | Proposed Feature Name | Description | Source | Type | Version | Notes |
|---|----------------------|-------------|--------|------|---------|-------|
| A1 | `customer_id` | Unique identifier (pass-through) | `customers.csv` | int | 1 | Row key, not a predictive feature |
| A2 | `acquisition_channel` | Marketing channel at signup | `customers.csv` | categorical (one-hot or embed) | 1 | 4 levels: organic, paid_search, referral, social |
| A3 | `region` | Geographic region | `customers.csv` | categorical (one-hot or embed) | 1 | 4 levels: east, north, south, west |
| A4 | `tenure_days` | Days since signup relative to as-of date | `customers.csv` | int | 1 | Derived: `as_of_date - signup_date`. Zero-activity default: 0 |
| A5 | `signup_year_month` | Year-month of signup (tenure bucket) | `customers.csv` | categorical | 1 | Captures seasonal cohort effects |

**Rationale:** Acquisition channel and region show churn rate variation (organic 15.3% vs paid_search 10.2%; east 16.4% vs north 12.3%). Tenure captures the idea that newer or much older customers may churn differently.

---

### Group B: Recency, Frequency, Monetary (RFM) Features (from `transactions.db` / `transactions_monthly_agg.parquet`)

Classic behavioral signals for customer churn, computed up to the as-of date.

| # | Proposed Feature Name | Description | Source | Type | Version | Zero-Activity Default |
|---|----------------------|-------------|--------|------|---------|----------------------|
| B1 | `recency_days` | Days since last transaction before as-of date | `transactions.db` | float | 1 | `tenure_days` (never purchased) |
| B2 | `total_txns` | Total transaction count before as-of date | `transactions.db` | int | 1 | 0 |
| B3 | `total_spend` | Sum of all transaction amounts before as-of date | `transactions.db` | float | 1 | 0.0 |
| B4 | `avg_txn_amount` | Average transaction amount before as-of date | `transactions.db` | float | 1 | 0.0 |
| B5 | `max_txn_amount` | Maximum single transaction amount | `transactions.db` | float | 1 | 0.0 |
| B6 | `txn_count_3m` | Transaction count in last 3 months before as-of | `transactions.db` | int | 1 | 0 |
| B7 | `txn_count_6m` | Transaction count in last 6 months before as-of | `transactions.db` | int | 1 | 0 |
| B8 | `spend_3m` | Total spend in last 3 months before as-of | `transactions.db` | float | 1 | 0.0 |
| B9 | `spend_6m` | Total spend in last 6 months before as-of | `transactions.db` | float | 1 | 0.0 |
| B10 | `return_count` | Total number of returns | `transactions.db` | int | 1 | 0 |
| B11 | `return_rate` | Return rate (returns ÷ total txns) | `transactions.db` | float | 1 | 0.0 |
| B12 | `return_count_3m` | Returns in last 3 months | `transactions.db` | int | 1 | 0 |
| B13 | `return_count_6m` | Returns in last 6 months | `transactions.db` | int | 1 | 0 |
| B14 | `months_active` | Number of distinct months with transaction activity | `transactions_monthly_agg.parquet` | int | 1 | 0 |
| B15 | `months_since_active` | Months since last active month | `transactions_monthly_agg.parquet` | int | 1 | `tenure_months` |

**Rationale:** RFM features are among the strongest predictors in churn models (see e.g., [Van den Poel & Larivière 2004](https://doi.org/10.1016/j.eswa.2003.10.005)). The time-windowed counts (3m/6m) capture recent behavioral trends, which matter more than lifetime aggregates for detecting changing engagement. Return rate captures dissatisfaction signals.

---

### Group C: Category & Channel Mix Features (from `transactions.db`)

Behavioral diversity and category preference signals.

| # | Proposed Feature Name | Description | Source | Type | Version | Zero-Activity Default |
|---|----------------------|-------------|--------|------|---------|----------------------|
| C1 | `distinct_categories` | Number of distinct product categories purchased | `transactions.db` | int | 1 | 0 |
| C2 | `distinct_channels` | Number of distinct purchase channels used | `transactions.db` | int | 1 | 0 |
| C3 | `preferred_channel` | Most-used purchase channel (mode) | `transactions.db` | categorical (one-hot) | 1 | `"mobile_app"` (default for zero-activity) |
| C4 | `mobile_app_ratio` | Fraction of transactions on mobile_app | `transactions.db` | float | 1 | 0.0 |
| C5 | `web_ratio` | Fraction of transactions on web | `transactions.db` | float | 1 | 0.0 |
| C6 | `store_ratio` | Fraction of transactions in-store | `transactions.db` | float | 1 | 0.0 |
| C7 | `category_entropy` | Shannon entropy of spend across categories (diversity measure) | `transactions.db` | float | 1 | 0.0 |
| C8 | `electronics_ratio` | Fraction of spend on electronics | `transactions.db` | float | 1 | 0.0 |
| C9 | `apparel_ratio` | Fraction of spend on apparel | `transactions.db` | float | 1 | 0.0 |
| C10 | `grocery_ratio` | Fraction of spend on grocery | `transactions.db` | float | 1 | 0.0 |

**Rationale:** Channel and category diversity signals whether a customer has broad engagement (harder to churn) or narrow reliance on one category/channel. The `mobile_app_ratio` is especially relevant — mobile users may have higher stickiness.

---

### Group D: Trend / Momentum Features (from `transactions_monthly_agg.parquet`)

Direction of change in customer behavior — is activity increasing or declining?

| # | Proposed Feature Name | Description | Source | Type | Version | Zero-Activity Default |
|---|----------------------|-------------|--------|------|---------|----------------------|
| D1 | `spend_trend_3m` | Slope of monthly spend over last 3 months | `transactions_monthly_agg.parquet` | float | 1 | 0.0 |
| D2 | `txn_count_trend_3m` | Slope of monthly txn count over last 3 months | `transactions_monthly_agg.parquet` | float | 1 | 0.0 |
| D3 | `spend_change_1m` | Month-over-month spend change (last month vs prev) | `transactions_monthly_agg.parquet` | float | 1 | 0.0 |
| D4 | `active_months_ratio` | Ratio of months active to months since signup | `transactions_monthly_agg.parquet` & `customers.csv` | float | 1 | 0.0 |

**Rationale:** A declining spend trend is a leading indicator of churn. If a customer's monthly spend has been dropping, they may be disengaging before they actually leave. These features capture the trajectory, not just the level.

---

### Group E: Derived Lifetime Value & Engagement (from `transactions.db`)

| # | Proposed Feature Name | Description | Source | Type | Version | Zero-Activity Default |
|---|----------------------|-------------|--------|------|---------|----------------------|
| E1 | `total_spend_monthly_avg` | Average spend per active month | `transactions_monthly_agg.parquet` | float | 1 | 0.0 |
| E2 | `txns_per_active_month` | Average transactions per active month | `transactions_monthly_agg.parquet` | float | 1 | 0.0 |
| E3 | `purchase_to_return_ratio` | Ratio of purchases to returns | `transactions.db` | float | 1 | `inf` (no returns = no lost spend) — cap at some large value |

**Rationale:** Per-active-month averages normalise for tenure, giving a "rate of engagement" independent of how long the customer has been around.

---

## Summary Table

| Group | Type | Count | Key Idea |
|-------|------|-------|----------|
| A | Static profile | 5 | Demographics, tenure, cohort |
| B | RFM + windows | 15 | Recency, frequency, monetary, returns, monthly activity |
| C | Category & channel mix | 10 | Diversity, channel preference, category allocation |
| D | Trend & momentum | 4 | Trajectory of spend/activity |
| E | Derived lifetime value | 3 | Per-month engagement rates |
| **Total** | | **37** | |

---

## Feature Engineering Constraints & Decisions

### As-of Date Semantics
- Every feature must be computed relative to an `as_of` date passed into `build_features()`.
- For training, `as_of` would be set per customer based on available churn date logic (or a single known cutoff).
- For scoring, `as_of` is the current timestamp.
- Only transactions with `txn_timestamp < as_of_date` contribute.

### Zero-Activity Handling
The ~81 customers with no transaction data (customer_ids present in CSV but absent from SQLite/parquet) receive:

| Feature Type | Default Value | Rationale |
|-------------|---------------|-----------|
| Count features (txns, returns, months) | 0 | No activity = zero counts |
| Sum / total features (spend) | 0.0 | No activity = zero spend |
| Average / ratio features (avg_txn, return_rate) | 0.0 | No activity → no meaningful ratio |
| Category / channel features (distinct_categories, ratio features) | 0.0 | No activity = no diversity |
| Channel preference | `"mobile_app"` | Arbitrary but known category for one-hot encoding; could also be all zeros |
| Trend features (slopes, deltas) | 0.0 | No data → no trend (stationary at zero) |
| Recency | `tenure_days` (signup to as-of) | Never purchased → recency equals tenure |
| Months since active | `tenure_months` | Never active → active gap equals tenure |

### Verifiable Feature Store
Every feature + version is stored as a parquet snapshot keyed by `customer_id`, with metadata in a SQLite registry. Lineage is documented in `FEATURE_LINEAGE.md`.

### Self-Review Notes for the Build Stage
The following should be reconsidered when implementing:
1. **Is `months_since_active` (B15) redundant with `recency_days` (B1)?** — Both measure "time since last engagement." Consider keeping only one. B1 has finer granularity (days), but B15 (months) smooths out noise. *Decision: keep both, different resolution.*
2. **Channel ratio features (C4–C6) may not all be needed** — They sum to 1.0, so any two plus a default for zero-activity capture the third. *Decision: keep all three for interpretability; model will handle collinearity.*
3. **The `purchase_to_return_ratio` (E3) has a division-by-zero edge case** — 0 returns → ratio is undefined. Cap at a large constant (e.g., `10 * max_ratio_in_data`). *Requires implementation care.*
4. **Category_entropy (C7) needs log(0) handling** — Any category with zero spend contributes 0 to entropy, which is correct (0 log 0 = 0 by convention), but must be explicit in code.

---

## Next Steps

1. Review and confirm this candidate feature set.
2. Formalize into `feature_spec.md` with exact signature, output type, version, and zero-activity default for each feature.
3. Implement `features/build.py`, `features/transforms.py`, `features/ingest.py`, and `features/schema.py`.
4. Build the feature store and run parity tests.

*Generated during Stage 0 of the Churn Feature Pipeline build process.*