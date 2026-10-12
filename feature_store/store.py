"""
store.py — Local feature store (SQLite registry + parquet value snapshots).

API:
  - register_version(feature_name, values_df, ...) -> version
  - get_latest(feature_name) -> DataFrame
  - get_version(feature_name, version) -> DataFrame
  - list_versions(feature_name) -> DataFrame
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import pandas as pd


REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "registry.db")
VALUES_DIR = os.path.join(os.path.dirname(__file__), "values")

# Tests redirect these to a temp directory instead of touching the real store.
# `set_store_location(...)` swaps both at once; production callers never need it.
_registry_path = REGISTRY_PATH
_values_dir = VALUES_DIR


def set_store_location(registry_path: str, values_dir: str) -> None:
    """Redirect registry + parquet storage to a different location (for tests)."""
    global _registry_path, _values_dir
    _registry_path = registry_path
    _values_dir = values_dir
    os.makedirs(values_dir, exist_ok=True)


def _resolve_absolute_parquet_path(parquet_path: str) -> str:
    """Resolve a stored relative parquet_path against the *active* values dir.

    When tests redirect the store to a temp directory, the relative parquet_path
    still starts with "feature_store/values/..."; re-anchor it to the active
    values dir so test snapshots are read back from where they were written.
    """
    if os.path.isabs(parquet_path):
        return parquet_path
    filename = os.path.basename(parquet_path)
    return os.path.join(_values_dir, filename)


def _get_connection() -> sqlite3.Connection:
    """Get a connection to the registry DB, creating tables if needed."""
    os.makedirs(os.path.dirname(_registry_path), exist_ok=True)
    conn = sqlite3.connect(_registry_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feature_registry (
            feature_name    TEXT NOT NULL,
            version         INTEGER NOT NULL,
            created_at      TEXT NOT NULL,
            description     TEXT NOT NULL,
            source_columns  TEXT NOT NULL,
            transform_summary TEXT NOT NULL,
            business_rationale TEXT NOT NULL,
            parameters      TEXT NOT NULL,
            parquet_path    TEXT NOT NULL,
            is_latest       INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (feature_name, version)
        )
    """)
    conn.commit()
    return conn


def register_version(
    feature_name: str,
    values_df: pd.DataFrame,
    description: str,
    source_columns: str,
    transform_summary: str,
    business_rationale: str,
    parameters: Optional[dict] = None,
) -> int:
    """Register a new version of a feature and return its version number.

    values_df must have columns customer_id, <feature_name>, as_of_date. The version number
    is allocated, the parquet snapshot written, the registry row inserted and is_latest
    flipped inside one IMMEDIATE transaction, so concurrent writers cannot allocate the same
    number. An existing parquet file is never overwritten (FileExistsError instead).
    """
    missing = {"customer_id", feature_name, "as_of_date"} - set(values_df.columns)
    if missing:
        raise ValueError(f"values_df for {feature_name} is missing columns: {sorted(missing)}")
    if values_df["as_of_date"].isnull().any():
        raise ValueError(f"values_df for {feature_name} has null as_of_date")

    os.makedirs(_values_dir, exist_ok=True)
    conn = _get_connection()
    conn.isolation_level = None  # manage the transaction explicitly
    parquet_file = None
    try:
        conn.execute("BEGIN IMMEDIATE")
        next_version = conn.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 FROM feature_registry WHERE feature_name = ?",
            (feature_name,),
        ).fetchone()[0]

        parquet_filename = f"{feature_name}__v{next_version}.parquet"
        parquet_file = os.path.join(_values_dir, parquet_filename)
        # Registry stores a repo-relative path; reads re-anchor it to the active values dir.
        rel_parquet_path = f"feature_store/values/{parquet_filename}"
        if os.path.exists(parquet_file):
            parquet_file = None  # not ours — must not be deleted on rollback
            raise FileExistsError(
                f"{parquet_filename} already exists but has no registry row; refusing to overwrite it"
            )
        values_df[["customer_id", feature_name, "as_of_date"]].to_parquet(parquet_file, index=False)

        conn.execute("UPDATE feature_registry SET is_latest = 0 WHERE feature_name = ?", (feature_name,))
        conn.execute(
            """INSERT INTO feature_registry
               (feature_name, version, created_at, description, source_columns,
                transform_summary, business_rationale, parameters, parquet_path, is_latest)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
            (
                feature_name, next_version, datetime.now(timezone.utc).isoformat(), description,
                source_columns, transform_summary, business_rationale,
                json.dumps(parameters or {}, sort_keys=True), rel_parquet_path,
            ),
        )
        conn.execute("COMMIT")
        return next_version
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        if parquet_file and os.path.exists(parquet_file):
            os.remove(parquet_file)  # only the file this call just wrote
        raise
    finally:
        conn.close()


def get_latest(feature_name: str) -> pd.DataFrame:
    """Get the highest registered version of a feature.

    Resolves via is_latest and cross-checks it against MAX(version), so a registry whose
    flag points at an older version fails loudly instead of serving stale values.
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT version, parquet_path FROM feature_registry WHERE feature_name = ? AND is_latest = 1",
            (feature_name,),
        ).fetchall()
        max_version = conn.execute(
            "SELECT MAX(version) FROM feature_registry WHERE feature_name = ?", (feature_name,)
        ).fetchone()[0]
    finally:
        conn.close()
    if max_version is None:
        raise KeyError(f"Feature '{feature_name}' has no registered versions.")
    if len(rows) != 1 or rows[0][0] != max_version:
        raise RuntimeError(
            f"Registry inconsistent for {feature_name}: is_latest rows {[r[0] for r in rows]}, max version {max_version}"
        )
    return pd.read_parquet(_resolve_absolute_parquet_path(rows[0][1]))


def get_version(feature_name: str, version: int) -> pd.DataFrame:
    """Get a specific version of a feature as a DataFrame."""
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT parquet_path FROM feature_registry WHERE feature_name = ? AND version = ?",
            (feature_name, int(version)),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise KeyError(f"Feature '{feature_name}' version {version} not found.")
    return pd.read_parquet(_resolve_absolute_parquet_path(row[0]))


def resolve_parquet_path(parquet_path: str) -> str:
    """Absolute on-disk location of a registry parquet_path (for checks and lineage docs)."""
    return _resolve_absolute_parquet_path(parquet_path)


def latest_parameters(feature_names: list[str]) -> dict[str, dict]:
    """{feature_name: parameters of its is_latest version} for features that have one."""
    conn = _get_connection()
    try:
        rows = conn.execute(
            f"SELECT feature_name, parameters FROM feature_registry WHERE is_latest = 1 "
            f"AND feature_name IN ({','.join('?' * len(feature_names))})",
            list(feature_names),
        ).fetchall()
    finally:
        conn.close()
    return {name: json.loads(params) for name, params in rows}


def list_feature_names() -> list[str]:
    conn = _get_connection()
    try:
        return [r[0] for r in conn.execute("SELECT DISTINCT feature_name FROM feature_registry ORDER BY 1")]
    finally:
        conn.close()


def list_versions(feature_name: str) -> pd.DataFrame:
    """List every registered version's metadata, most recent first."""
    conn = _get_connection()
    try:
        df = pd.read_sql(
            """SELECT feature_name, version, created_at, description, source_columns,
                      transform_summary, business_rationale, parameters, parquet_path, is_latest
               FROM feature_registry
               WHERE feature_name = ?
               ORDER BY version DESC""",
            conn,
            params=(feature_name,),
        )
        return df
    finally:
        conn.close()