# Artifact Notes — self-critique log

This build resumed an earlier, interrupted run (see `current_state.md`). That run had produced the code and a
populated feature store; the resumed run re-verified Step 0, re-read every module against `feature_spec.md`, and
found that the previous run's "done" items were not all correct (§1).

**Revision 2026-10-12: training as_of moved to 2026-05-03.** The learner revised `feature_spec.md` and
`data_dictionary.md` (not edited here) and cleared `feature_store/registry.db` and `feature_store/values/`.
This revision changed the training default, updated the tests and checks that assumed 1,500 customers / 81
zero-activity, rebuilt the store from scratch, and regenerated the docs. See §0 and §4.

## 0. Label leakage at 2026-08-01: resolved by the spec revision

At as_of = 2026-08-01 the rule **`has_txn_history == 0 or days_since_last_txn >= 90`** reproduced `churned` for
**1,499 of 1,500 customers** (the exception, customer 697, sits on the 90-day boundary). `churned` appears to be
defined as "no purchase in the 90 days before 2026-08-01", so recency and 90-day features at that date
restated the label.

The spec now trains at **2026-05-03**, 90 days before 2026-08-01, so features use only data from before the
label window. Verified on the rebuilt store (`VERSION_COMPARISON.md` computes this from the snapshots):

- The same rule matches `churned` for **1,000 of 1,400** eligible customers. Always predicting "not churned"
  scores 1,245 / 1,400, so the rule is now *worse* than a constant: the features no longer restate the label.
- The training default lives only in `scripts/build_training_features.py` (`TRAINING_AS_OF = "2026-05-03"`),
  as the spec's parity section requires. It was previously `features/schema.py::TRAINING_AS_OF` and was also
  the default for `score_customer.py`. Scoring now **requires** `--as-of` ("for scoring it is the date passed
  in"), so a live score can no longer silently use the training date.

## 1. Bugs found in the previous run's code and fixed (history)

| # | Where | What was wrong | Fix |
|---|---|---|---|
| 1 | `features/ingest.py` | **Leakage.** as_of filter was a SQL *string* comparison `txn_timestamp < '2026-05-01T00:00:00'`, but the table stores `'2026-05-01 00:00:05.…'` (space separator, which sorts before `T`), so every row on the as_of day leaked in. At as_of 2026-05-01: 8,308 rows instead of 8,269. Invisible at 2026-08-01 only because there is no data on/after that date; **it would have leaked at the new 2026-05-03 training date.** | Filter on parsed timestamps; transforms also re-apply `< as_of` (defence in depth). Regression test `test_sqlite_loader_excludes_the_as_of_day`. |
| 2 | `features/transforms.py` `spend_ratio_last_3m_vs_previous` | Window ended on the customer's last *non-returned purchase*; spec says "the 90 days ending on the customer's last transaction". Also returned the 1.0 default when the recent window had no purchase but the previous period had spend; the spec's computation gives 0 there. | Both ratios share one helper anchored on the last transaction of any kind. In the old store this was registered as `spend_ratio` v2. **After the 2026-05-03 rebuild the store holds only v1, which already carries the fix**; `REVISION_NOTES` in the build script is empty. |
| 3 | Window boundaries | `return_count_recent` used `(last−N, last]`, the ratios used `[last−90, last]`: two conventions for one spec phrase. | One convention, `(last_txn − N, last_txn]`, documented in `schema.py` and `FEATURE_LINEAGE.md`, boundary-tested. |
| 4 | Feature store snapshots | Every old v1/v2 parquet had `as_of_date = None`; `life_span_days` v1 was float64 though the schema says int64. | `register_version` rejects snapshots without `as_of_date`. **The legacy files are gone** (store cleared and rebuilt): every snapshot now records `as_of_date = 2026-05-03` and the schema dtype. |
| 5 | `feature_store/store.py` | Version number read before the transaction began; `to_parquet` could silently overwrite an orphan file; `get_latest` trusted `is_latest` blindly. | Allocation + write + insert + `is_latest` flip inside one `BEGIN IMMEDIATE`; existing file → `FileExistsError`; `get_latest` cross-checks `MAX(version)`. |
| 6 | `scripts/score_customer.py` | **Train/score skew.** Scoring used each versioned feature's *v1* params while the store's latest are v2. | Params resolved from the store's latest versions; refuses without them. |
| 7 | `features/quality_checks.py` standalone drift | Same skew: `--drift` built with v1 params vs store-latest v2 → false PSI 0.94. | Builds with store-latest params. |
| 8 | `features/quality_checks.py` | `check_bounds` kept one failure per feature; no duplicate-customer check; null channel short-circuited; not runnable standalone. | All fixed; `python -m features.quality_checks --as-of …`. |
| 9 | Zero-activity ids in the build script | Derived from *all* transactions, not those before as_of. **Would have been wrong at 2026-05-03** (81 vs 230). | `zero_activity_ids_from_raw(txns, ids, as_of)`, independent of `has_txn_history`. |
| 10 | `scripts/compare_feature_versions.py` | "Net change" was a count difference, not crossings; all-pairs; hard-coded list. | Per-customer crossings, consecutive versions, list from `schema.VERSIONED_FEATURES`. |
| 11 | `scripts/build_training_features.py` | Re-running registered a new version of every feature each run. | Idempotent. **Re-verified after the rebuild:** a second run registers nothing. |
| 12 | Population | Customers who signed up on/after as_of were included at historical dates (negative `days_since_last_txn` default). | `signup_date < as_of`. No effect at 2026-08-01; **at 2026-05-03 it excludes 100 customers** (1,400 built), which the spec's row-count check now requires. |

## 2. Re-verifications (rerun at 2026-05-03)

- **Zero-activity default**: on the real build all **230** customers with no transaction before 2026-05-03
  carry exactly 0 / 1.0 / 0.0 and `days_since_last_txn` = days since signup (quality check + tests).
- **as_of threading**: `build_features` raises if as_of is None; every transform applies `as_of`; the only
  default is the training script's argument.
- **`get_latest` returns the highest version**: tested with N = 3 and a corrupted `is_latest`. Real store:
  11 registry rows, 11 parquet files, every `is_latest` row is its feature's max version.
- **Spec figures checked against the data** (all match): 1,400 customers with `signup_date < 2026-05-03`,
  churn 11.1%; 230 with no transaction before it; 44.2% joined in the 12 months before; label rule 1,499/1,500
  at 2026-08-01 and 1,000/1,400 at 2026-05-03; return churn 9.8% (any return) vs 10.6% (transacting, no
  return); median days since last purchase among transacting customers 24, median 90-day count 2; 99th
  percentile of the 90-day return count 1; median ratios among transacting customers 1.10 (spend) and 1.45
  (count). All values inside the spec bounds (max `life_span_days` 802, spend ratio 233.8, txn ratio 48.8).

## 3. Left as-is, deliberately

- **The parquet is loaded by nothing in the build**, per spec. Its loader remains in `ingest.py` and is tested.
- **`tests/test_transforms.py::test_loaders_read_real_sources` still asserts 1,500 rows.** It checks the raw
  `customers.csv` file size, not the feature population, and that is still 1,500.
- **The synthetic fixture (`tests/conftest.py`) keeps `AS_OF = 2026-08-01`.** Its hand-worked expected values
  are tied to that date; it does not touch the real data.

## 4. Findings from this revision for the learner (spec / dictionary not edited)

1. **Channel churn rates in the spec are full-population figures.** `acquisition_channel`'s rationale cites
   10.2% (paid search) vs 15.3% (organic) over all 1,500 customers. Over the 1,400 in the training population
   they are **7.9% vs 12.9%**. The direction holds; the figures are for a different population than the one
   being trained on.
2. **The versioning rationales are not supported at the new as_of.** In `VERSION_COMPARISON.md`:
   - `return_count_recent`: v2 (180 d) was chosen to "separate customers better", but univariate AUC goes
     0.501 → 0.499 and flagged/unflagged churn is 10.9% vs 11.1%. Neither window separates churners.
   - `txn_count_window`: v1 (90 d) AUC 0.441, v2 (30 d) 0.476, so the 90-day count is the one further from
     chance at this date. This is single-date and univariate, not a model result, but the v2 rationale ("shows
     a recent slowdown sooner") is not visible in it.
3. **`data_dictionary.md` §3** still says July 2026 "matters most for features calculated at
   `as_of = 2026-08-01`". At the 2026-05-03 training date July is not read at all; it now matters for scoring at
   the current date (as `feature_spec.md`'s multi-source section already says).
4. **Scoring at 2026-08-01 vs the 2026-05-03 training snapshots** (`python -m features.quality_checks --as-of
   2026-08-01 --drift`): passes, max PSI 0.153 (`life_span_days`: customers have had three more months of
   life span). The PSI for `has_txn_history` reads 0.0 although the zero share falls from 16% to 5%; see
   `review_log.md` E1 (drift-check bug, not fixed in this revision).
