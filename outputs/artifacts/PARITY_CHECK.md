# Parity Check

Train/score parity verification for `features/build.py::build_features(as_of, source_paths, customer_ids, params)`.
The only difference between the training call (`customer_ids=None`) and the scoring call
(`customer_ids=[...]`) is the final row filter.

Regenerated 2026-10-12 after the training as_of moved to **2026-05-03** (`feature_spec.md`). The feature store
was rebuilt from scratch at that date (1,400 customers with `signup_date < as_of`, 230 with no transaction
before it), and the mutation check below was rerun against the updated suite.

## What `tests/test_parity.py` asserts

| Test | What it drives |
|---|---|
| `test_training_and_scoring_paths_agree_on_real_data` ×4 | Real data, as_of ∈ {2026-05-03 (training, imported from `scripts/build_training_features.py`), 2026-08-01 (the day after the latest transaction, i.e. what scoring "today" sees)} × {first, last spec'd version params}. The training batch is compared (exact equality, `check_exact=True`) against (1) one scoring-path call covering every customer and (2) individual single-customer scoring calls for ~60 customers: everyone with raw rows on or just after as_of, zero-activity, shortest tenure, heaviest, top returners, lowest spend ratio, plus a fixed random sample. |
| `test_training_and_scoring_paths_agree_on_boundary_fixture` | Synthetic sources written as real CSV/SQLite/parquet files with rows placed *exactly* on every window edge (as_of−30d, as_of−90d, last_txn−90d, last_txn−180d, as_of), every customer scored individually. |
| `test_scoring_with_store_params_reproduces_the_registered_latest_snapshots` | End to end: `score_customer.py`'s parameter resolution (latest store version) reproduces the values `build_training_features.py` registered, scored at each snapshot's own `as_of_date`. Now **exact** equality with matching dtypes: the rebuilt store has no legacy float64 / null-`as_of_date` snapshots, so the old 1e-12 tolerance was removed. Customer 104 was replaced by 17 because 104 signed up after 2026-05-03. |
| `test_as_of_is_required` | as_of is never defaulted inside the feature module. |
| `test_no_feature_reads_the_monthly_parquet` | Added by the coverage check below. |

## 1. Parity mutation check

Each mutation was applied to a scratch copy of the repo (session scratchpad, not this repo) by injecting a
scoring-only branch (`if customer_ids is not None:`) into `features/build.py`; the full suite was then run
in that copy. The real `features/build.py` was never modified.

| Mutation (scoring path only) | Result (rerun 2026-10-12) |
|---|---|
| **M1**: duplicate `days_since_last_txn` that rounds days instead of flooring | **Caught**: 5 tests fail (all 4 real-data parity cases + the store-snapshot test). |
| **M2**: duplicate `txn_count_window` with an exclusive window start (`>` instead of `>=`) | **Caught**, but **only** by the boundary-fixture test (1 failure). The real data's microsecond timestamps never sit exactly on a window edge, so the real-data parity tests alone would have missed it. This is why the boundary fixture exists. |
| **M3**: scoring path loads transactions with the old SQL string-compare as_of filter (leaks the whole as_of day) | **Not caught, and correctly so.** Every transform re-applies `txn_timestamp < as_of` itself, so the leaked rows never reach a feature: a full-population build at 2026-05-03 via the mutated scoring path was verified identical to the training build (`DataFrame.equals` → True). |
| **M3b**: M3 plus removing that in-transform as_of guard from `compute_days_since_last_txn` (so the leak *does* reach a feature) | **Caught**: 6 tests fail (both real-data parity cases at as_of 2026-05-03, the boundary-fixture parity test, 3 transform tests). |

Moving the training as_of from 2026-05-01 to 2026-05-03 did not weaken M3b: 2026-05-03 also has raw rows on
the as_of day, so the leak reaches real customers and the real-data parity cases catch it.

After the mutation runs, the real repo's suite: **57 passed** (`outputs/artifacts/pytest_output.txt`).

## 2. Coverage-of-intent check

`feature_spec.md` items vs. tests that exercise them:

| Spec item | Covered by |
|---|---|
| `life_span_days`, `days_since_last_txn`, `has_txn_history`, `return_pct`, `acquisition_channel` | `test_transforms.py` normal / zero-activity / single-record cases |
| `return_count_recent` (window ends on last txn), both spec'd windows | normal case (90 and 500 days), `test_return_count_recent_boundaries`, parity at 90 and 180 |
| `txn_count_window`, both spec'd windows | normal case (30, 90), `test_txn_count_window_boundaries`, parity at 90 and 30 |
| `spend_ratio_*` (non-returned spend; window ends on last txn of any kind; default 1.0 cases) | normal case, `test_spend_ratio_window_ends_on_last_transaction_of_any_kind`, `test_single_purchase`, `test_single_return`, `test_ratios_default_when_no_previous_period` |
| `txn_ratio_*` | normal case, customer 6 case, `test_ratio_recent_window_boundary`, no-previous-period case |
| No transaction on/after as_of used | `test_rows_on_or_after_as_of_are_ignored`, `test_sqlite_loader_excludes_the_as_of_day`, zero-activity customer 5 |
| Parity design (one path, as_of threaded, training default set only in the training script) | `test_parity.py`; the default lives only in `scripts/build_training_features.py` (`TRAINING_AS_OF`), and `score_customer.py --as-of` is now required |
| Versioning: 2 features × 2 versions, every version retrievable, `is_latest` = highest | `test_feature_store.py` (temp store + read-only checks of the real store) |
| DQ: no nulls | `test_null_check_fires_on_injected_null`, `..._on_missing_column` |
| DQ: zero-activity defaults (incl. days since signup) | `test_zero_activity_check_fires_on_blanket_fillna_zero`, `test_zero_activity_days_since_check_fires_on_wrong_default` |
| DQ: every bound in the spec, incl. channel | `test_outlier_check_fires_on_out_of_range_value` (8 cases), `test_channel_check_fires_on_unknown_or_missing_value` |
| DQ: one row per customer; row count = customers with `signup_date < as_of` for training | `test_row_count_check_fires_on_duplicated_and_dropped_customers` (expected count derived from the data at the training as_of, no longer a hardcoded 1,500) |
| Spec figures: 1,400 eligible / 230 zero-activity at 2026-05-03 | `test_population_matches_the_figures_stated_in_the_spec` (new) |
| Store snapshots built at the training as_of over exactly the eligible population | `test_real_store_snapshots_are_built_at_the_training_as_of_for_the_eligible_population` (new) |
| DQ: failure exits non-zero | `test_assert_all_checks_pass_exits_non_zero` |

**Gaps found and closed (tests added):**
1. *Row count must match the requested `customer_ids` on the scoring path*: spec'd, previously untested →
   `test_quality_checks.py::test_row_count_check_matches_requested_ids_on_the_scoring_path`.
2. *No feature reads `transactions_monthly_agg.parquet`* (spec multi-source section): previously untested →
   `test_parity.py::test_no_feature_reads_the_monthly_parquet` (build is identical with the parquet path pointing at a non-existent file).
3. *(this revision)* *Population = `signup_date < as_of`*: the old tests hardcoded 1,500 rows / 81 zero-activity,
   which silently assumed as_of was after every signup. Now derived from the data, plus the two new tests above.

The drift check (`check_drift`, PSI) is beyond the spec's list; it is tested
(`test_drift_check_fires_on_shifted_distribution`, `test_drift_check_tolerates_small_noise`) and run by
`python -m features.quality_checks --as-of ... --drift`.
