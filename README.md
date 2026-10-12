# Churn Feature Pipeline

A feature pipeline for the e-commerce churn dataset. It builds 9 churn features from three
sources, stores them in a local feature store (SQLite registry + parquet snapshots), keeps
more than one version of two features, and checks that training and scoring give the same
values.

What the features are, where each one comes from, and why, is in `feature_spec.md`. The data
sources are described in `data_dictionary.md`.

## Requirements

- Python 3.10 or later
- The packages in `requirements.txt` (pandas, numpy, pyarrow, pytest)

Everything runs locally. No credentials or environment variables are needed.

## Setup

From the project root.

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks the activate script, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` first.

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Keep the virtual environment active for all the commands below.

## Data

The three raw files go in `data/` and are not edited:

- `data/customers.csv`: one row per customer, including the `churned` label
- `data/transactions.db`: SQLite, table `transactions`, one row per transaction
- `data/transactions_monthly_agg.parquet`: monthly aggregate. It is loaded but not used for
  any feature, because it is behind `transactions.db` for July 2026.

## Running the scripts

Run the scripts as modules (`python -m scripts.<name>`). Running the `.py` file directly
fails with `ModuleNotFoundError`.

### 1. Build the training features

```powershell
python -m scripts.build_training_features
```

- Builds all features at `as_of = 2026-05-03` (the training date; see `feature_spec.md` for
  why it is not 2026-08-01).
- Runs the data-quality checks and stops with a non-zero exit if any check fails.
- Registers every feature in `feature_store/registry.db` and writes one snapshot per version
  to `feature_store/values/`.
- Running it again does not register anything new if the values have not changed.

### 2. Score a customer

```powershell
python -m scripts.score_customer --customer-id 2 --as-of 2026-05-03
```

- `--as-of` is required.
- Uses the same `build_features` function as training, with the parameters of the latest
  version of each feature in the store.
- For the same customer and date, the values match the training snapshot in the store.

### 3. Compare feature versions

```powershell
python -m scripts.compare_feature_versions
```

Loads every version of each versioned feature from the store and writes
`VERSION_COMPARISON.md`.

## Running the tests

```powershell
python -m pytest -v
```

The tests cover the transforms, train/score parity, the data-quality checks and the feature
store. Run the training build first, because some tests read the real feature store.

## Checking the feature store by hand

```powershell
python -c "import sqlite3, pandas as pd; print(pd.read_sql('SELECT feature_name, version, is_latest, parquet_path FROM feature_registry ORDER BY feature_name, version', sqlite3.connect('feature_store/registry.db')).to_string())"
```

Every feature should have at least one version, exactly one row per feature should have
`is_latest = 1`, and every `parquet_path` should exist.

## Project layout

```
data/                 raw sources (not edited)
features/             ingest.py, transforms.py, schema.py, quality_checks.py, build.py
feature_store/        store.py, registry.db, values/*.parquet
scripts/              build_training_features.py, score_customer.py, compare_feature_versions.py
tests/                test_transforms.py, test_parity.py, test_quality_checks.py, test_feature_store.py
outputs/artifacts/    FEATURE_LINEAGE.md, VERSION_COMPARISON.md, PARITY_CHECK.md,
                      ARTIFACT_NOTES.md, review_log.md, pytest output and other evidence
feature_spec.md       the confirmed feature spec
data_dictionary.md    the reviewed data dictionary
```

## Versioned features

| Feature | Parameter | v1 | v2 |
|---|---|---|---|
| `txn_count_window` | `window_days` | 90 | 30 |
| `return_count_recent` | `window_days` | 90 | 180 |

The reason for each version is in `feature_spec.md` and `outputs/artifacts/FEATURE_LINEAGE.md`.