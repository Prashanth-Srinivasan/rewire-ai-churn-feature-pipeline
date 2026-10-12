# Review Log — Step 3 self-review

Reviewed as a first-time reader: `features/` (ingest, transforms, schema, build, quality_checks),
`feature_store/store.py`, `scripts/` (build_training_features, score_customer, compare_feature_versions,
generate_lineage), `tests/`, and the generated docs. Focus areas per the skill: silent train/score
divergence, null/default handling vs. the spec, and silent overwrites in the feature store.

Status key: **Fixed** (change made, test noted) · **Left** (reason given). No finding is left unaddressed.

Sections A–D are the original review (statuses updated where the 2026-10-12 revision changed them). Section E
covers the revision that moved the training as_of to 2026-05-03 and rebuilt the store from scratch.

## A. Train/score divergence

| # | Finding | Status |
|---|---|---|
| A1 | `score_customer.py` scored with `build_features`' default params (v1 windows) while the store's latest versions use v2 windows — a model trained from the store would be scored on different feature definitions. | **Fixed** — params come from `store.latest_parameters`; refuses without them. `test_scoring_with_store_params_reproduces_the_registered_latest_snapshots`. |
| A2 | Same skew in `quality_checks --drift` (built v1, compared to v2 → false PSI 0.94). | **Fixed** — `test_standalone_runner_passes_on_real_data`. |
| A3 | Parity test sampled customers randomly; a scoring-only leak reaching a feature could slip through. | **Fixed** — full-population scoring call + customers with rows at as_of always included (see PARITY_CHECK.md). |
| A4 | Boundary-only divergences are invisible on real data (microsecond timestamps). | **Fixed** — boundary fixture parity test; mutation M2 is caught only by it. |
| A5 | `build_features` still has one structural risk: `params` can differ between call sites. | **Left (by design)** — the spec allows parameters per version; both scripts now derive them from one source (spec for training registration, store for scoring) and the end-to-end test pins them together. |
| A6 | Scripts did `os.chdir(ROOT)` at import time; `test_parity.py` imports `scripts.score_customer`, so importing changed the test process's cwd. | **Fixed** — chdir moved into each `main()`. |

## B. Null / default handling vs. the spec

| # | Finding | Status |
|---|---|---|
| B1 | `spend_ratio` window anchored on last purchase, and 1.0 instead of 0 when recent spend is zero but previous spend exists. | **Fixed** — v2 registered with reason; `test_spend_ratio_window_ends_on_last_transaction_of_any_kind`. |
| B2 | Ratio "no previous period" rule: previous period = `[signup_date, last_txn − 90d]`; when that span is ≤ 0 days or holds nothing, default 1.0. Matches spec text. | **Verified** — `test_ratios_default_when_no_previous_period`, `test_single_purchase`. |
| B3 | Very short previous periods (e.g. 1 day) produce extreme ratios (a single row in 1 day = 30/month baseline). Spec-faithful; current max 250 (spend) / 51 (txn), inside the 500 / 100 bounds. | **Left** — spec computation; the bounds check will fail loudly if future data exceeds it. Worth a spec note (minimum previous-period length) — raised to learner here, not changed. |
| B4 | `acquisition_channel` has no default and nulls are not masked — a null must fail a check. A null was first only reported by `no_nulls` because `run_all_checks` short-circuited. | **Fixed** — short-circuit only on missing columns; `test_channel_check_fires_on_unknown_or_missing_value`. |
| B5 | `life_span_days` / `days_since_last_txn` use floor of whole days (`.dt.days`); spec says "days". | **Left** — explicit convention, documented in schema and tested (`test_days_since_last_txn_floors_partial_days`). |
| B6 | Zero-activity ids used by checks were derived from all transactions, not those before as_of, and partly from the same logic as `has_txn_history`. | **Fixed** — `zero_activity_ids_from_raw`, independent of the feature. |

## C. Feature store overwrites / integrity

| # | Finding | Status |
|---|---|---|
| C1 | `to_parquet` would silently overwrite an orphan `<feature>__v<n>.parquet`. | **Fixed** — `FileExistsError`, registry untouched; `test_registering_never_overwrites_an_existing_snapshot`. |
| C2 | Version allocated outside the write transaction. | **Fixed** — `BEGIN IMMEDIATE` covers allocate → write → insert → flip. |
| C3 | `get_latest` trusted `is_latest` blindly. | **Fixed** — cross-checks `MAX(version)`; `test_get_latest_fails_loudly_if_is_latest_is_inconsistent`; N=3 test. |
| C4 | Re-running the training build re-registered everything, shifting version numbers. | **Fixed** — idempotent registration; verified twice (no new rows on rerun). |
| C5 | Snapshots lacked `as_of_date`. | **Fixed** — enforced for every registration; the legacy files were removed when the learner cleared the store, and all 11 rebuilt snapshots record `as_of_date = 2026-05-03`. `test_real_store_snapshots_are_built_at_the_training_as_of_for_the_eligible_population`. |
| C6 | Feature-store tests must not touch the real registry. | **Verified** — temp store fixture restores module paths; real registry still 11 rows / 11 files after the suite. |
| C7 | `PRAGMA journal_mode=WAL` leaves `registry.db-wal/-shm` beside the DB while connections are open. | **Left** — harmless for a local single-user store; connections are closed after each call. |

## D. Maintainability / other

| # | Finding | Status |
|---|---|---|
| D1 | Feature definitions were duplicated across schema, build script rationale strings, and docs. | **Fixed** — `schema.py` holds the spec-derived computation/default/rationale and the versioning plan; registry text and `FEATURE_LINEAGE.md` are generated from it. |
| D2 | `compare_feature_versions.py` crossing metric was a count difference, not crossings. | **Fixed**. |
| D3 | `ingest.load_transactions_sqlite` loads the whole table and filters in pandas. | **Left** — 11,742 rows; correctness (format-independent filtering) beats a SQL prefilter here. Revisit if the table grows by orders of magnitude. |
| D4 | Training as_of 2026-08-01 makes the label a function of recency features (1,499/1,500 match). | **Fixed (spec revised by the learner)** — training default is now 2026-05-03; the rule matches 1,000/1,400, worse than predicting "not churned" for everyone (1,245/1,400). ARTIFACT_NOTES §0. |
| D5 | `return_count_recent` v2 (180d) was chosen to separate churners better. At 2026-08-01 it did not (AUC 0.468 → 0.451). | **Left — spec decision.** The revised spec now calls returns an unsupported hypothesis. At 2026-05-03: AUC 0.501 → 0.499, flagged 10.9% vs unflagged 11.1% churn, so neither window separates churners. The v2 version reason ("separates customers better") is still not borne out; reported in VERSION_COMPARISON.md and ARTIFACT_NOTES §4. |

## E. Revision 2026-10-12 — training as_of 2026-05-03, store rebuilt from scratch

| # | Finding | Status |
|---|---|---|
| E1 | **Drift check is blind to binary / low-cardinality numeric features.** `population_stability_index` bins on reference quantiles, then sets the outer edges to ±inf; for `has_txn_history` (0/1) the edges collapse to a single bin, so PSI is always 0.0. Scoring at 2026-08-01 vs the 2026-05-03 snapshots: zero-activity share falls from 16% to 5%, PSI reported 0.0. | **Left — outside this revision's scope, recommended fix:** treat numeric features with ≤ ~10 distinct reference values as categorical (bin by value). A shift of that size would score ≈ 0.14, still under the 0.25 threshold, so no current check result would change. |
| E2 | Tests hardcoded 1,500 rows and 81 zero-activity customers, which silently assumed every customer had signed up before as_of and that "zero-activity" meant no transactions ever. | **Fixed** — `test_quality_checks.py` derives the expected count from `signup_date < as_of` and the zero-activity set from raw rows before as_of; the real-store test checks each snapshot holds exactly the eligible customers; one explicit test pins the spec's stated 1,400 / 230. |
| E3 | The training as_of was `features/schema.py::TRAINING_AS_OF` and was also `score_customer.py`'s default, so a scoring call without `--as-of` would silently score at the training date. Spec: the default is set only in the training script's arguments; scoring uses the date passed in. | **Fixed** — `TRAINING_AS_OF` lives in `scripts/build_training_features.py`; tests and `generate_lineage.py` import it from there; `score_customer.py --as-of` is required. |
| E4 | The store-snapshot parity test compared with a 1e-12 tolerance and ignored dtype, to accommodate legacy float64 snapshots. With the legacy files gone that slack could hide a real dtype or rounding change. | **Fixed** — exact equality and matching dtypes, scored at each snapshot's own `as_of_date`. |
| E5 | `VERSION_COMPARISON.md`'s leakage caveat was a hardcoded string about 2026-08-01. | **Fixed** — `compare_feature_versions.py` computes population, churn rate and the label-rule match from the stored snapshots. |
| E6 | `schema.py` rationale text had drifted from the spec: the return rationales were reworded in the spec, and the `txn_count_window` v2 reason said "26 days" where the spec now says 24 (24 verified at 2026-05-03). These texts feed the registry and lineage. | **Fixed** before the rebuild, so every registry row carries the current spec text. |
| E7 | Customer 104 (used by two tests) signed up 2026-06-26, after the training as_of. | **Fixed** — replaced by customer 17. |
| E8 | `REVISION_NOTES` still described the old spend_ratio v1→v2 fix ("59 of 1,500 customers change"); after the rebuild there is no v1 to revise. | **Fixed** — emptied, with a comment; the history stays in ARTIFACT_NOTES §1 #2. |
| E9 | Spec's channel churn figures (10.2% / 15.3%) are over all 1,500 customers; over the 1,400 in the training population they are 7.9% / 12.9%. `data_dictionary.md` §3 still ties July 2026's importance to as_of 2026-08-01. | **Left — fixed inputs, not edited.** Raised in ARTIFACT_NOTES §4. |

Final state after this revision: store rebuilt (11 registry rows / 11 parquet files, 1,400 rows each,
`as_of_date = 2026-05-03`); rerunning `build_training_features.py` registers nothing new; `pytest` → **57 passed**;
`python -m features.quality_checks --as-of 2026-05-03 --drift` → all pass; parity mutation check rerun with the
same outcomes (PARITY_CHECK.md).
