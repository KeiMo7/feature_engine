"""Tests for diffing two feature snapshots."""

import pandas as pd
import pytest

from feature_engine.creation import MathFeatures
from feature_engine.pipeline import Pipeline
from feature_engine.preprocessing import RenameFeatures
from feature_engine.selection import DropFeatures
from feature_engine.snapshot import FeatureSnapshot, diff_snapshots


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [5.0, 6.0, 7.0, 8.0],
            "x3": [9.0, 10.0, 11.0, 12.0],
        }
    )


def _v1(df):
    return Pipeline(
        [
            ("math", MathFeatures(variables=["x1", "x2"], func=["sum", "mean"])),
            ("rename", RenameFeatures(rename_dict={"x3": "x3_renamed"})),
            ("drop", DropFeatures(features_to_drop=["x2"])),
        ]
    ).fit(df)


def _v2(df):
    # same chain, but the sum feature is renamed and a brand-new max feature
    # is added, while the mean feature is dropped.
    return Pipeline(
        [
            (
                "math",
                MathFeatures(variables=["x1", "x2"], func=["sum", "mean", "max"]),
            ),
            (
                "rename",
                RenameFeatures(
                    rename_dict={
                        "x3": "x3_renamed",
                        "sum_x1_x2": "total_x1_x2",
                    }
                ),
            ),
            ("drop", DropFeatures(features_to_drop=["x2", "mean_x1_x2"])),
        ]
    ).fit(df)


def test_diff_detects_rename_as_single_identity(df):
    # Acceptance: a rename must be recognised as the same feature, not as a
    # remove plus an add.
    old_snapshot = _v1(df).create_feature_snapshot(df)
    new_snapshot = _v2(df).create_feature_snapshot(df)

    diff = diff_snapshots(old_snapshot, new_snapshot)

    assert diff.renamed == [
        {
            "identity": "derive::math_features.MathFeatures::x1+x2",
            "old_name": "sum_x1_x2",
            "new_name": "total_x1_x2",
        }
    ]
    # the renamed feature must not also appear as added or removed
    assert "total_x1_x2" not in diff.added
    assert "sum_x1_x2" not in diff.removed


def test_diff_separates_added_and_removed(df):
    old_snapshot = _v1(df).create_feature_snapshot(df)
    new_snapshot = _v2(df).create_feature_snapshot(df)
    diff = diff_snapshots(old_snapshot, new_snapshot)

    assert diff.added == ["max_x1_x2"]
    assert diff.removed == ["mean_x1_x2"]
    # the four categories are independently populated and never overlap
    all_changed = (
        diff.added
        + diff.removed
        + [item["new_name"] for item in diff.renamed]
        + [item["name"] for item in diff.dtype_changed]
    )
    assert len(all_changed) == len(set(all_changed))


def test_diff_detects_dtype_change(df):
    old = Pipeline([("math", MathFeatures(variables=["x1", "x2"], func="sum"))]).fit(df)
    old_snapshot = old.create_feature_snapshot(df)

    df_int = df.copy()
    df_int["x1"] = df_int["x1"].astype("int64")
    df_int["x2"] = df_int["x2"].astype("int64")
    new = Pipeline([("math", MathFeatures(variables=["x1", "x2"], func="sum"))]).fit(
        df_int
    )
    new_snapshot = new.create_feature_snapshot(df_int)

    diff = diff_snapshots(old_snapshot, new_snapshot)
    changed = {item["name"]: item for item in diff.dtype_changed}
    assert "sum_x1_x2" in changed
    assert changed["sum_x1_x2"]["old_dtype"] == "float64"
    assert changed["sum_x1_x2"]["new_dtype"] == "int64"


def test_diff_detects_parameter_change(df):
    old_snapshot = _v1(df).create_feature_snapshot(df)
    new_snapshot = _v2(df).create_feature_snapshot(df)
    changed_steps = {
        item["step"]: item
        for item in diff_snapshots(old_snapshot, new_snapshot).step_changes
    }
    assert "math" in changed_steps
    assert changed_steps["math"]["change"] == "params_changed"


def test_identical_snapshots_have_empty_diff(df):
    snapshot = _v1(df).create_feature_snapshot(df)
    diff = diff_snapshots(snapshot, FeatureSnapshot(snapshot.to_dict()))
    assert diff.is_empty
    assert diff.to_dict() == {
        "added": [],
        "removed": [],
        "renamed": [],
        "dtype_changed": [],
        "step_changes": [],
    }


def test_diff_accepts_plain_dicts(df):
    old = _v1(df).create_feature_snapshot(df).to_dict()
    new = _v2(df).create_feature_snapshot(df).to_dict()
    diff = diff_snapshots(old, new)
    assert diff.renamed[0]["new_name"] == "total_x1_x2"


def test_pipeline_level_diff_helper(df):
    old_snapshot = _v1(df).create_feature_snapshot(df)
    new_pipeline = _v2(df)
    diff = new_pipeline.diff_feature_snapshots(old_snapshot)
    assert diff.renamed[0]["new_name"] == "total_x1_x2"
    assert diff.added == ["max_x1_x2"]
