"""Tests for rollback and snapshot corruption / version handling."""

import json

import pandas as pd
import pytest

from feature_engine.creation import MathFeatures
from feature_engine.pipeline import Pipeline
from feature_engine.preprocessing import RenameFeatures
from feature_engine.selection import DropFeatures
from feature_engine.snapshot import (
    FeatureSnapshot,
    SnapshotCorruptionError,
    SnapshotVersionError,
    restore_pipeline,
    rollback,
)


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [5.0, 6.0, 7.0, 8.0],
            "x3": [9.0, 10.0, 11.0, 12.0],
        }
    )


@pytest.fixture
def snapshot(df):
    pipe = Pipeline(
        [
            ("math", MathFeatures(variables=["x1", "x2"], func=["sum", "mean"])),
            ("rename", RenameFeatures(rename_dict={"x3": "x3_renamed"})),
            ("drop", DropFeatures(features_to_drop=["x2"])),
        ]
    ).fit(df)
    return pipe.create_feature_snapshot(df, name="v1")


# --------------------------------------------------------------------------- #
# Rollback
# --------------------------------------------------------------------------- #
def test_restore_pipeline_rebuilds_unfitted_chain(snapshot):
    restored = restore_pipeline(snapshot)
    assert isinstance(restored, Pipeline)
    assert [name for name, _ in restored.steps] == [
        "math",
        "rename",
        "drop",
    ]
    # restored pipeline starts unfitted
    for _, step in restored.steps:
        assert not hasattr(step, "feature_names_in_")


def test_rollback_regenerates_identical_feature_set(snapshot, df):
    # Acceptance: after rollback the feature set equals the snapshot exactly.
    restored = rollback(snapshot, df, verify=True)
    output = restored.transform(df)

    assert output.columns.tolist() == snapshot.feature_names
    # and the restored chain's own feature-name view agrees too
    assert restored.get_feature_names_out() == snapshot.feature_names


def test_rollback_values_match_original_chain(snapshot, df):
    original = Pipeline(
        [
            ("math", MathFeatures(variables=["x1", "x2"], func=["sum", "mean"])),
            ("rename", RenameFeatures(rename_dict={"x3": "x3_renamed"})),
            ("drop", DropFeatures(features_to_drop=["x2"])),
        ]
    ).fit(df)
    restored = rollback(snapshot, df, verify=True)

    pd.testing.assert_frame_equal(restored.transform(df), original.transform(df))


def test_rollback_from_file_path(tmp_path, snapshot, df):
    path = snapshot.to_json_file(tmp_path / "snap.json")
    restored = rollback(path, df, verify=True)
    assert restored.transform(df).columns.tolist() == snapshot.feature_names


def test_rollback_without_data_returns_unfitted(snapshot):
    restored = rollback(snapshot)
    assert isinstance(restored, Pipeline)
    for _, step in restored.steps:
        assert not hasattr(step, "feature_names_in_")


def test_verify_true_raises_on_mismatch(snapshot, df):
    payload = snapshot.to_dict()
    # tamper with the expected feature names
    for feature in payload["features"]:
        if feature["name"] == "x3_renamed":
            feature["name"] = "tampered"
    broken = FeatureSnapshot(payload)
    with pytest.raises(AssertionError):
        rollback(broken, df, verify=True)


# --------------------------------------------------------------------------- #
# Corruption
# --------------------------------------------------------------------------- #
def test_invalid_json_reports_file_segment(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(SnapshotCorruptionError) as excinfo:
        FeatureSnapshot.from_json_file(path)
    assert excinfo.value.segment == "file"
    assert "line" in str(excinfo.value)


def test_missing_section_reports_segment(tmp_path, snapshot):
    payload = snapshot.to_dict()
    del payload["steps"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SnapshotCorruptionError) as excinfo:
        FeatureSnapshot.from_json_file(path)
    assert excinfo.value.segment == "root"
    assert "steps" in str(excinfo.value)


def test_malformed_step_reports_which_step(tmp_path, snapshot):
    payload = snapshot.to_dict()
    del payload["steps"][1]["class_path"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SnapshotCorruptionError) as excinfo:
        FeatureSnapshot.from_json_file(path)
    assert excinfo.value.segment == "steps[1]"


def test_duplicate_live_name_reports_feature_segment(tmp_path, snapshot):
    payload = snapshot.to_dict()
    payload["features"][1]["name"] = payload["features"][0]["name"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SnapshotCorruptionError) as excinfo:
        FeatureSnapshot.from_json_file(path)
    assert excinfo.value.segment.startswith("features[")


def test_unsupported_format_version(tmp_path, snapshot):
    payload = snapshot.to_dict()
    payload["format_version"] = "99.0"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SnapshotCorruptionError) as excinfo:
        FeatureSnapshot.from_json_file(path)
    assert excinfo.value.segment == "format_version"


# --------------------------------------------------------------------------- #
# Version migration
# --------------------------------------------------------------------------- #
def test_missing_class_gives_migration_advice(snapshot):
    payload = snapshot.to_dict()
    payload["steps"][0]["class_path"] = "feature_engine.does_not_exist.Ghost"
    broken = FeatureSnapshot(payload)
    with pytest.raises(SnapshotVersionError) as excinfo:
        restore_pipeline(broken)
    assert "migration advice" in str(excinfo.value).lower()
    assert excinfo.value.step_name == "math"


def test_rejected_parameters_give_migration_advice(snapshot):
    payload = snapshot.to_dict()
    # Point the step at a real class but feed it an invalid constructor arg.
    payload["steps"][0]["params"] = {"variables": ["x1"], "bogus_param": 1}
    broken = FeatureSnapshot(payload)
    with pytest.raises(SnapshotVersionError):
        restore_pipeline(broken)
