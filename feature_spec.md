# feature_spec.md — Churn Feature Pipeline

## Candidate features

Notes that apply to every feature below:
- `churned` is the label. It is never used as an input. No transaction on or after `as_of`
  is used.
- Do not use `transactions_monthly_agg.parquet` for any feature, because it is behind for
  July 2026. Every aggregation is redone from the `transactions` table in
  `data/transactions.db`.
- "Last 3 months" means the 90 days ending on the customer's last transaction before
  `as_of`. "Previous period" means from `signup_date` up to the start of those 90 days. For
  monthly averages, a month is 30 days.
- Spend is the sum of `amount` for rows with `is_returned = 0`.
- No "active months" feature is included. 44% of eligible customers joined in the 12 months
  before the training `as_of`, so a count of active months would be limited by how new the
  customer is for too many of them.
- Training `as_of` is 2026-05-03, not 2026-08-01. At 2026-08-01 the label matched "no
  purchase in the 90 days before 2026-08-01" for 1,499 of 1,500 customers, so `churned`
  appears to be defined on that window, and recency and 90-day features at that date would
  restate the label. 2026-05-03 is 90 days before 2026-08-01, so features use only data
  before the label window. At 2026-05-03 the match drops to 1,000 of 1,400.
- Only customers with `signup_date < as_of` are built: 1,400 at the training `as_of`, of
  whom 230 have no transaction before it.

- **Feature**: `life_span_days`
- **Source**: `data/transactions.db` (`txn_timestamp`), `data/customers.csv` (`signup_date`)
- **Computation**: days from `signup_date` to the customer's last transaction before `as_of`
- **Zero-activity default**: 0
- **If multiple sources could supply this, which and why?**: `transactions.db`, it has the
  exact date of the last transaction.
- **Rationale**: This predicts churn because customers who stopped buying soon after signup
  tend to be one-time buyers who do not come back, distinct from just low spend/frequency
  because it measures how long the customer stayed active, not how much they bought.

- **Feature**: `return_count_recent` (versioned)
- **Source**: `data/transactions.db` (`is_returned`, `txn_timestamp`)
- **Computation**: number of rows with `is_returned = 1` in the last `window_days` days
  ending on the customer's last transaction before `as_of`
- **Zero-activity default**: 0
- **If multiple sources could supply this, which and why?**: `transactions.db`, the window
  ends on the customer's own last transaction date, not on a calendar month.
- **Rationale**: This predicts churn because customers who returned items in their most
  recent months may leave after a bad experience, distinct from just low spend/frequency
  because a customer can be buying normally and still be unhappy with what they receive.
  This is a hypothesis the data does not yet support: at the training `as_of`, customers with
  any return churned at 9.8% against 10.6% without. I keep the feature so a model can test
  it, and because returns can also mean an engaged customer.

- **Feature**: `spend_ratio_last_3m_vs_previous`
- **Source**: `data/transactions.db` (`amount`, `is_returned`, `txn_timestamp`),
  `data/customers.csv` (`signup_date`)
- **Computation**: average monthly spend in the last 3 months divided by average monthly
  spend in the previous period
- **Zero-activity default**: 1.0. The same value is used when there is no previous period or
  no spend in it. This is a neutral placeholder; the usual value in the data is about 1.1,
  because the last 3 months always end on a purchase.
- **If multiple sources could supply this, which and why?**: `transactions.db`, both parts
  of the ratio must come from the same complete data.
- **Rationale**: This predicts churn because customers whose recent spend has fallen below
  their own earlier average tend to be reducing their business before leaving, distinct from
  just low spend because a customer who always spent little gets a usual value, and only a
  real drop against their own past gives a low value.

- **Feature**: `txn_ratio_last_3m_vs_previous`
- **Source**: `data/transactions.db` (`txn_timestamp`), `data/customers.csv` (`signup_date`)
- **Computation**: average monthly transaction count in the last 3 months divided by average
  monthly transaction count in the previous period
- **Zero-activity default**: 1.0. The same value is used when there is no previous period or
  no transaction in it. This is a neutral placeholder; the usual value in the data is about
  1.45, because the last 3 months always end on a purchase.
- **If multiple sources could supply this, which and why?**: `transactions.db`, both parts
  of the ratio must come from the same complete data.
- **Rationale**: This predicts churn because customers who were ordering less often than
  their own earlier pace tend to be losing interest, distinct from just low frequency
  because a customer who always ordered rarely gets a usual value, and only a real slowdown
  against their own past gives a low value.

- **Feature**: `acquisition_channel`
- **Source**: `data/customers.csv` (`acquisition_channel`)
- **Computation**: kept as a text category (`organic`, `paid_search`, `referral`, `social`)
- **Zero-activity default**: not needed, it does not depend on transactions. A missing or
  unknown value fails the quality check.
- **If multiple sources could supply this, which and why?**: only `customers.csv` has it.
- **Rationale**: This predicts churn because customers who came through paid search were
  already looking for a product and tend to stay (10.2% churn against 15.3% for organic),
  distinct from just low spend/frequency because the channel is fixed at signup.

- **Feature**: `has_txn_history`
- **Source**: `data/transactions.db` (`customer_id`, `txn_timestamp`)
- **Computation**: 1 if the customer has at least one transaction before `as_of`, else 0
- **Zero-activity default**: 0
- **If multiple sources could supply this, which and why?**: `transactions.db`, the parquet
  is missing 10 customers who do have transactions.
- **Rationale**: This predicts churn because customers who signed up and never bought tend
  to drift away without any event, distinct from just low spend/frequency because it
  separates "never started" from "started and slowed down".

- **Feature**: `days_since_last_txn`
- **Source**: `data/transactions.db` (`txn_timestamp`), `data/customers.csv` (`signup_date`)
- **Computation**: days from the customer's last transaction before `as_of` to `as_of`.
  For training `as_of` is 2026-05-03 (see the notes at the top); for scoring it is the date
  passed in.
- **Zero-activity default**: days from `signup_date` to `as_of`
- **If multiple sources could supply this, which and why?**: `transactions.db`, I need the
  exact day and the parquet only has the month.
- **Rationale**: This predicts churn because customers who have not bought for a long time
  tend to have moved elsewhere, distinct from just low spend/frequency because a customer
  with many past orders can still have a long gap now.

- **Feature**: `txn_count_window` (versioned)
- **Source**: `data/transactions.db` (`txn_timestamp`)
- **Computation**: number of transactions in the last `window_days` days before `as_of`
- **Zero-activity default**: 0
- **If multiple sources could supply this, which and why?**: `transactions.db`, the window
  is in days and ends on `as_of`.
- **Rationale**: This predicts churn because customers who are ordering less often right now
  tend to be disengaging, distinct from just low spend because it counts visits and not
  money. It shows the current level of activity, while `txn_ratio_last_3m_vs_previous` shows
  the change against the customer's own past.

- **Feature**: `return_pct`
- **Source**: `data/transactions.db` (`is_returned`, `txn_timestamp`)
- **Computation**: rows with `is_returned = 1` divided by all rows before `as_of`, times 100
- **Zero-activity default**: 0.0
- **If multiple sources could supply this, which and why?**: `transactions.db`, the parquet
  return counts are incomplete for July 2026.
- **Rationale**: This predicts churn because customers who return a large share of what they
  buy may stop ordering, distinct from just low spend/frequency because it is a share, so a
  regular buyer with many returns is flagged and an occasional buyer with none is not. As
  with `return_count_recent`, the data shows almost no difference in churn for customers with
  returns, so this is kept as a hypothesis for the model to test.

## Train/score parity design

- **Shared function/code path**: `build_features(as_of, source_paths=None,
  customer_ids=None, params=None)` in `features/build.py`
- **Only allowed training/scoring difference**: which `customer_ids` are selected at the end
- **How as_of is threaded (not hardcoded)**: always passed as a parameter to every
  transform; the training default (2026-05-03) is set only in the script's arguments

## Feature store, versioning & lineage

- **Features versioned**: `txn_count_window`, `return_count_recent`
- **Feature**: `txn_count_window` / **Parameter**: `window_days` / **Value per version**:
  v1 = 90, v2 = 30 / **Reason each version differs**: the usual customer buys about twice in
  90 days and last bought 24 days ago, so a 90-day count still looks normal for someone who
  has been silent for two months; 30 days shows a recent slowdown sooner
- **Feature**: `return_count_recent` / **Parameter**: `window_days` / **Value per version**:
  v1 = 90, v2 = 180 / **Reason each version differs**: returns are rare, so in a 90-day
  window the count is 0 for almost every customer (99th percentile is 1); 180 days picks up
  more returns and separates customers better
- **What forces a version bump vs. not**: a change in parameter, source, computation or
  default; not a rename or a reworded description
- **What a lineage entry must let someone reconstruct**: source → row filter and window →
  computation → default → parameters → rationale → `parquet_path`

## Multi-source judgment calls

Do not use `transactions_monthly_agg.parquet` directly for any feature. My data check showed
it is behind the `transactions` table for July 2026, the most recent month: 102
customer-months are missing, 110 more have different counts and amounts, and 237
transactions (about 2%) are not included. Months up to June 2026 match. Training at
2026-05-03 would not see July, but scoring at the current date would, and July is the most
recent month for it. Using one source for both keeps training and scoring identical.
Any aggregation that is needed (by customer, by month, transaction count, total amount,
return count) must be redone from the `transactions` table in `data/transactions.db`.
`features/ingest.py` can still have a loader for the parquet, but no transform reads it.

- **Feature**: `txn_count_window`, `txn_ratio_last_3m_vs_previous` — **Source chosen**:
  `transactions.db` — **Why**: day-level windows, and the parquet undercounts July 2026.
  Stage 0 proposed the parquet for transaction trends; I changed it.
- **Feature**: `spend_ratio_last_3m_vs_previous` — **Source chosen**: `transactions.db` —
  **Why**: the parquet totals are incomplete for July 2026 and cannot exclude returned rows.
  Stage 0 proposed the parquet for spend trends; I changed it.
- **Feature**: `return_count_recent`, `return_pct` — **Source chosen**: `transactions.db`
  — **Why**: the parquet return counts are incomplete for July 2026.
- **Feature**: `life_span_days`, `days_since_last_txn`, `has_txn_history` — **Source
  chosen**: `transactions.db` — **Why**: exact last transaction date, and all 1,419
  transacting customers are present (10 of them are missing from the parquet).

## Data-quality checks

- **Check**: no nulls in any feature after defaults are applied — **Catches**: a join or
  reindex that leaves the zero-transaction customers (230 at the training `as_of`) as NaN —
  **On failure**: raise /
  non-zero exit
- **Check**: customers with no transactions before `as_of` have exactly the stated defaults
  (0 for counts, `life_span_days` and `return_pct`; 1.0 for the two ratios; days since
  signup for `days_since_last_txn`) — **Catches**: a blanket `fillna(0)` that sets the ratio
  and recency defaults wrongly — **On failure**: raise / non-zero exit
- **Check**: `spend_ratio_last_3m_vs_previous` between 0 and 500;
  `txn_ratio_last_3m_vs_previous` between 0 and 100; `return_pct` between 0 and 100;
  `return_count_recent` between 0 and 10; `txn_count_window` between 0 and 50;
  `life_span_days` between 0 and 900; `days_since_last_txn` not negative;
  `acquisition_channel` is one of the four known values — **Catches**: duplicated
  transaction rows inflating counts, a window that ignores `as_of`, transactions dated
  before signup, or a new channel value coming in unnoticed — **On failure**: raise /
  non-zero exit
- **Check**: one row per customer, and the row count matches the number of customers with
  `signup_date < as_of` for training (1,400 at 2026-05-03) or the requested `customer_ids`
  for scoring — **Catches**: a join with the
  transactions that repeats customers, or an inner join that drops the zero-transaction
  customers — **On failure**: raise / non-zero exit