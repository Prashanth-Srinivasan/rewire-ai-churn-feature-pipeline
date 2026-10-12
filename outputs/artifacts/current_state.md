# Current State of the Churn Feature Pipeline Build

**Status: complete** (revised 2026-10-12). Training as_of moved to **2026-05-03** per the revised
`feature_spec.md`; the feature store was rebuilt from scratch at that date and every generated doc regenerated.
History of the earlier runs and fixes: `ARTIFACT_NOTES.md`, `review_log.md`.

| Step | Status | Artifacts |
|---|---|---|
| 0 — data loads, DQ realities confirmed | ✅ | `outputs/artifacts/step0_data_checks_rerun.txt` |
| 1a — feature module | ✅ | `features/*.py` |
| 1b — feature store | ✅ 11 registry rows / 11 parquet files, 1,400 rows each, as_of_date 2026-05-03 | `feature_store/`, `outputs/artifacts/build_training_features_2026-05-03.txt` |
| 1c — versioning exercise | ✅ `txn_count_window` v1=90→v2=30, `return_count_recent` v1=90→v2=180 | `VERSION_COMPARISON.md` |
| 1d — lineage + self-critique | ✅ | `FEATURE_LINEAGE.md` (generated), `ARTIFACT_NOTES.md` |
| 2 — tests + parity verification | ✅ 57 passed | `tests/`, `PARITY_CHECK.md`, `outputs/artifacts/pytest_output.txt` |
| 3 — self-review | ✅ | `review_log.md` |

**Resolved:** the label leakage at 2026-08-01 (1,499/1,500 match). At 2026-05-03 the recency rule matches
1,000/1,400, worse than a constant "not churned" prediction.

**Open items for the learner** (details in `ARTIFACT_NOTES.md` §4 and `review_log.md` §E): the versioning
rationales are not borne out at 2026-05-03; the spec's channel churn figures are full-population; the drift
check reads PSI 0 for binary features.

Reproduce:
```
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # (bin/ on macOS/Linux)
python scripts/build_training_features.py                 # default as_of 2026-05-03
python scripts/compare_feature_versions.py
python scripts/generate_lineage.py
python -m features.quality_checks --as-of 2026-05-03 --drift
python scripts/score_customer.py --customer-id 2 8 --as-of 2026-05-03   # --as-of is required
pytest
```
