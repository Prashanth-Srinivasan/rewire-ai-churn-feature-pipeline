"""
Feature store tests. Registration tests run against a temporary registry + values dir
(never the real store); the last tests inspect the real feature_store/ read-only.

For each feature feature_spec.md designates as versioned: register every spec'd version
with its real values, confirm each is independently retrievable by version number, that
is_latest / get_latest point at the highest version (also with N > 2), that list_versions
returns every version with distinct parquet paths that exist, and that nothing is overwritten.
"""

import json
import os
import sqlite3

import pandas as pd
import pytest

from feature_store import store
from features import ingest
from features.build import build_features
from features.schema import FEATURES, VERSIONED_FEATURES
from scripts.build_training_features import TRAINING_AS_OF

AS_OF = pd.Timestamp(TRAINING_AS_OF)
REPO_REGISTRY = os.path.join("feature_store", "registry.db")


@pytest.fixture
def temp_store(tmp_path):
    saved = (store._registry_path, store._values_dir)
    store.set_store_location(str(tmp_path / "registry.db"), str(tmp_path / "values"))
    yield tmp_path
    store._registry_path, store._values_dir = saved


def _values(feature, params):
    df = build_features(AS_OF, params={feature: params})[["customer_id", feature]]
    return df.assign(as_of_date=AS_OF.date().isoformat())


def _register(feature, params, n=None):
    return store.register_version(
        feature, _values(feature, params), description=f"{feature} test", source_columns="data/transactions.db",
        transform_summary="test", business_rationale=f"params {params}", parameters=params,
    )


@pytest.mark.parametrize("vf", VERSIONED_FEATURES, ids=lambda vf: vf.name)
def test_register_and_retrieve_all_versions_of_a_versioned_feature(temp_store, vf):
    expected = {}
    for i, v in enumerate(vf.versions, start=1):
        assert _register(vf.name, v.parameters) == i
        expected[i] = _values(vf.name, v.parameters)
    for version, values in expected.items():
        got = store.get_version(vf.name, version)
        assert list(got.columns) == ["customer_id", vf.name, "as_of_date"]
        pd.testing.assert_frame_equal(got, values.reset_index(drop=True))
    # the spec'd versions really differ (otherwise the version bump would be cosmetic)
    assert not expected[1][vf.name].equals(expected[len(expected)][vf.name])


@pytest.mark.parametrize("vf", VERSIONED_FEATURES, ids=lambda vf: vf.name)
def test_is_latest_points_to_highest_version_not_an_earlier_one(temp_store, vf):
    for v in vf.versions:
        _register(vf.name, v.parameters)
    # a third version, re-using v1's parameters, must become latest: get_latest follows
    # registration order / version number, not parameter values or file names
    third = _register(vf.name, vf.versions[0].parameters)
    meta = store.list_versions(vf.name)
    assert third == len(vf.versions) + 1
    assert meta.loc[meta["is_latest"] == 1, "version"].tolist() == [third]
    pd.testing.assert_frame_equal(store.get_latest(vf.name), store.get_version(vf.name, third))
    assert store.latest_parameters([vf.name]) == {vf.name: vf.versions[0].parameters}


def test_list_versions_returns_distinct_parquet_paths_that_exist(temp_store):
    vf = VERSIONED_FEATURES[0]
    for v in vf.versions:
        _register(vf.name, v.parameters)
    meta = store.list_versions(vf.name)
    assert meta["version"].tolist() == list(range(len(vf.versions), 0, -1))  # most recent first
    assert meta["parquet_path"].is_unique
    assert all(os.path.exists(store.resolve_parquet_path(p)) for p in meta["parquet_path"])
    assert [json.loads(p) for p in meta["parameters"][::-1]] == [v.parameters for v in vf.versions]


def test_registering_never_overwrites_an_existing_snapshot(temp_store):
    vf = VERSIONED_FEATURES[0]
    _register(vf.name, vf.versions[0].parameters)
    v1_path = store.resolve_parquet_path(store.list_versions(vf.name).iloc[0]["parquet_path"])
    v1_bytes = open(v1_path, "rb").read()
    _register(vf.name, vf.versions[1].parameters)
    assert open(v1_path, "rb").read() == v1_bytes

    # An orphan file where v3 would go (e.g. a crashed earlier run) is refused, not overwritten,
    # and the registry is left unchanged.
    orphan = os.path.join(store._values_dir, f"{vf.name}__v3.parquet")
    with open(orphan, "wb") as f:
        f.write(b"orphan")
    with pytest.raises(FileExistsError):
        _register(vf.name, vf.versions[0].parameters)
    assert open(orphan, "rb").read() == b"orphan"
    assert len(store.list_versions(vf.name)) == 2
    assert store.list_versions(vf.name).iloc[0]["is_latest"] == 1


def test_register_rejects_snapshot_without_as_of_date(temp_store):
    df = _values("txn_count_window", {"window_days": 30}).drop(columns="as_of_date")
    with pytest.raises(ValueError):
        store.register_version("txn_count_window", df, "d", "s", "t", "r", {})
    assert store.list_versions("txn_count_window").empty


def test_get_latest_fails_loudly_if_is_latest_is_inconsistent(temp_store):
    vf = VERSIONED_FEATURES[0]
    for v in vf.versions:
        _register(vf.name, v.parameters)
    with sqlite3.connect(store._registry_path) as conn:
        conn.execute("UPDATE feature_registry SET is_latest = CASE version WHEN 1 THEN 1 ELSE 0 END")
    with pytest.raises(RuntimeError):
        store.get_latest(vf.name)


def test_unknown_feature_or_version_raises(temp_store):
    with pytest.raises(KeyError):
        store.get_latest("nope")
    with pytest.raises(KeyError):
        store.get_version("nope", 1)


# ---------------------------------------------------------------- the real store (read-only)

def _real_registry():
    with sqlite3.connect(REPO_REGISTRY) as conn:
        return pd.read_sql("SELECT * FROM feature_registry", conn)


def test_real_store_has_every_candidate_feature_with_one_correct_latest():
    reg = _real_registry()
    assert {f.name for f in FEATURES} <= set(reg["feature_name"])
    for name, rows in reg.groupby("feature_name"):
        latest = rows[rows["is_latest"] == 1]
        assert len(latest) == 1 and int(latest["version"].iloc[0]) == int(rows["version"].max()), name
        assert rows["parquet_path"].is_unique
        assert all(os.path.exists(p) for p in rows["parquet_path"]), name


def test_real_store_snapshots_are_built_at_the_training_as_of_for_the_eligible_population():
    """Every snapshot holds one row per customer with signup_date < as_of (spec row-count check)."""
    customers = ingest.load_customers_csv("data/customers.csv")
    eligible = set(customers.loc[customers["signup_date"] < AS_OF, "customer_id"])
    reg = _real_registry()
    for _, row in reg.iterrows():
        snap = store.get_version(row["feature_name"], int(row["version"]))
        assert snap["as_of_date"].unique().tolist() == [AS_OF.date().isoformat()], row["feature_name"]
        assert snap["customer_id"].is_unique and set(snap["customer_id"]) == eligible, row["feature_name"]


def test_real_store_holds_every_spec_version_of_each_versioned_feature():
    reg = _real_registry()
    assert len(VERSIONED_FEATURES) >= 2
    for vf in VERSIONED_FEATURES:
        rows = reg[reg["feature_name"] == vf.name].sort_values("version")
        params = [json.loads(p) for p in rows["parameters"]]
        assert len(rows) >= 2
        for v in vf.versions:
            assert v.parameters in params, (vf.name, v.parameters)
        assert params[-1] == vf.versions[-1].parameters  # latest carries the spec's final value
        values = [store.get_version(vf.name, int(v))[vf.name] for v in rows["version"]]
        assert not values[0].equals(values[-1]), vf.name  # the spec'd versions really differ
