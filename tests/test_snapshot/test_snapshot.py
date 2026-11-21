"""Tests for feature snapshots: creation, structure and lineage tracing."""

import json

import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from feature_engine.creation import MathFeatures
from feature_engine.pipeline import Pipeline
from feature_engine.preprocessing import RenameFeatures
from feature_engine.selection import DropFeatures
from feature_engine.snapshot import (
    FeatureSnapshot,
    FeatureNameConflictError,
    build_snapshot,
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
def pipeline(df):
    pipe = Pipeline(
        [
            ("math", MathFeatures(variables=["x1", "x2"], func=["sum", "mean"])),
            ("rename", RenameFeatures(rename_dict={"x3": "x3_renamed"})),
            ("drop", DropFeatures(features_to_drop=["x2"])),
        ]
    )
    return pipe.fit(df)


# --------------------------------------------------------------------------- #
# 1. Normal snapshot
# --------------------------------------------------------------------------- #
def test_snapshot_matches_pipeline_output(pipeline, df):
    snapshot = pipeline.create_feature_snapshot(df)

    # alive feature names and order equal the real transformed output
    assert snapshot.feature_names == pipeline.transform(df).columns.tolist()
    assert snapshot.feature_names == [
        "x1",
        "x3_renamed",
        "sum_x1_x2",
        "mean_x1_x2",
    ]


def test_snapshot_contains_structure_but_no_data_rows(pipeline, df):
    snapshot = pipeline.create_feature_snapshot(df)
    payload = snapshot.to_dict()

    # structural sections present
    assert set(
        [
            "format_version",
            "created_at",
            "library",
            "environment",
            "input_features",
            "steps",
            "features",
        ]
    ).issubset(payload.keys())

    # every step carries class path and parameters, never data
    for step in payload["steps"]:
        assert "class_path" in step and "params" in step
        assert "data" not in step and "values" not in step

    # the raw cell values of the input never appear anywhere in the file
    text = json.dumps(payload)
    assert "9.0" not in text and "10.0" not in text
    assert "11.0" not in text and "12.0" not in text


def test_snapshot_records_dtypes(df):
    pipe = Pipeline([("math", MathFeatures(variables=["x1", "x2"], func="sum"))]).fit(
        df
    )
    snapshot = pipe.create_feature_snapshot(df)
    record = snapshot.feature("sum_x1_x2")
    assert record["dtype"] == "float64"
    assert record["birth_step"] == "math"
    assert record["dependencies"] == ["x1", "x2"]


def test_snapshot_lineage_traces_back_to_inputs(pipeline, df):
    snapshot = pipeline.create_feature_snapshot(df)
    lineage = snapshot.lineage("mean_x1_x2")

    assert lineage["birth_step"] == "math"
    assert lineage["dependencies"] == ["x1", "x2"]
    actions = [event["action"] for event in lineage["history"]]
    assert actions[0] == "created"
    assert "passthrough" in actions


def test_snapshot_records_dropped_feature_history(pipeline, df):
    snapshot = pipeline.create_feature_snapshot(df)
    x2 = next(f for f in snapshot.features if f["name"] == "x2")
    assert x2["alive"] is False
    assert x2["history"][-1]["action"] == "dropped"
    assert x2["history"][-1]["step"] == "drop"


def test_metadata_only_snapshot_touches_no_data(pipeline, df):
    snapshot = pipeline.create_feature_snapshot()
    assert snapshot.feature_names == pipeline.transform(df).columns.tolist()
    # dtypes cannot be observed without running data
    assert all(feature["dtype"] is None for feature in snapshot.features)


def test_snapshot_json_roundtrip(tmp_path, pipeline, df):
    snapshot = pipeline.create_feature_snapshot(df, name="v1")
    path = snapshot.to_json_file(tmp_path / "snap.json")
    reloaded = FeatureSnapshot.from_json_file(path)
    assert reloaded.to_dict() == snapshot.to_dict()


def test_snapshot_is_deterministic_across_runs(pipeline, df):
    first = pipeline.create_feature_snapshot(df).to_dict()
    second = pipeline.create_feature_snapshot(df).to_dict()
    # only the wall-clock timestamp should differ
    first.pop("created_at")
    second.pop("created_at")
    assert first == second


def test_snapshot_requires_fitted_pipeline(df):
    pipe = Pipeline([("math", MathFeatures(variables=["x1", "x2"], func="sum"))])
    with pytest.raises(ValueError):
        pipe.create_feature_snapshot(df)


# --------------------------------------------------------------------------- #
# 5. Name conflicts
# --------------------------------------------------------------------------- #
class RecreateColumn(TransformerMixin, BaseEstimator):
    """Drop x1 then emit another column also named x1."""

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = X.shape[1]
        self.variables_ = ["x1"]
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(columns=["x1"])
        X["x1"] = 99.0
        return X


def test_same_name_recreation_raises_conflict(df):
    pipe = Pipeline([("recreate", RecreateColumn())]).fit(df)
    with pytest.raises(FeatureNameConflictError) as excinfo:
        pipe.create_feature_snapshot(df)
    # the error names both producers
    assert "input" in str(excinfo.value)
    assert "recreate" in str(excinfo.value)
    assert "x1" in str(excinfo.value)


class DuplicateLabel(TransformerMixin, BaseEstimator):
    """Emit a dataframe containing a duplicated column label."""

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        frame = pd.DataFrame({c: X[c] for c in X.columns} | {"dup": X["x1"]})
        return pd.concat([frame, X[["x1"]].rename(columns={"x1": "dup"})], axis=1)


def test_duplicate_output_label_raises_conflict(df):
    pipe = Pipeline([("dup", DuplicateLabel())]).fit(df)
    with pytest.raises(FeatureNameConflictError) as excinfo:
        pipe.create_feature_snapshot(df)
    assert "dup" in str(excinfo.value)
    assert excinfo.value.feature_name == "dup"


# --------------------------------------------------------------------------- #
# Identity stability internals
# --------------------------------------------------------------------------- #
def test_identity_is_based_on_lineage_not_name(pipeline, df):
    snapshot = pipeline.create_feature_snapshot(df)
    renamed = snapshot.feature("x3_renamed")
    # a renamed root keeps a root:: identity keyed by the original column
    assert renamed["identity"] == "root::x3"
    assert renamed["birth_step"] is None


def test_build_snapshot_helper_returns_plain_dict(pipeline, df):
    payload = build_snapshot(pipeline, df)
    assert isinstance(payload, dict)
    validated = FeatureSnapshot(payload)
    assert validated.feature_names == pipeline.transform(df).columns.tolist()


def test_explicit_renames_parameter_keeps_identity(df):
    # A third-party-style transformer that renames without feature_renames_:
    # the explicit renames= mapping must make the snapshot keep one identity.
    class ManualRename(TransformerMixin, BaseEstimator):
        def fit(self, X, y=None):
            self.feature_names_in_ = X.columns.tolist()
            self.n_features_in_ = X.shape[1]
            return self

        def transform(self, X):
            return X.rename(columns={"x3": "x3_renamed"})

    pipe = Pipeline([("manual", ManualRename())]).fit(df)
    snapshot = pipe.create_feature_snapshot(
        df, renames={"manual": {"x3": "x3_renamed"}}
    )
    renamed = snapshot.feature("x3_renamed")
    assert renamed["identity"] == "root::x3"
    assert renamed["history"][-1]["action"] == "renamed"


def test_bad_explicit_rename_raises(df):
    class ManualRename(TransformerMixin, BaseEstimator):
        def fit(self, X, y=None):
            self.feature_names_in_ = X.columns.tolist()
            self.n_features_in_ = X.shape[1]
            return self

        def transform(self, X):
            return X.rename(columns={"x3": "x3_renamed"})

    pipe = Pipeline([("manual", ManualRename())]).fit(df)
    with pytest.raises(ValueError):
        pipe.create_feature_snapshot(df, renames={"manual": {"missing": "x3_renamed"}})


def test_final_predictor_step_is_not_recorded(df):
    from sklearn.linear_model import LinearRegression

    pipe = Pipeline(
        [
            ("math", MathFeatures(variables=["x1", "x2"], func="sum")),
            ("model", LinearRegression()),
        ]
    ).fit(df, df["x1"])
    snapshot = pipe.create_feature_snapshot(df)
    assert [step["name"] for step in snapshot.steps] == ["math"]
    assert "model" not in [step["name"] for step in snapshot.steps]


def test_row_removing_transformer_does_not_break_tracing(df):
    import numpy as np
    from feature_engine.imputation import DropMissingData

    df_na = df.copy()
    df_na.loc[0, "x1"] = np.nan
    y = pd.Series([1, 2, 3, 4])  # target without missing values
    pipe = Pipeline(
        [
            ("dropna", DropMissingData()),
            ("math", MathFeatures(variables=["x1", "x2"], func="sum")),
        ]
    ).fit(df_na, y)
    snapshot = pipe.create_feature_snapshot(df_na)
    assert snapshot.feature_names == pipe.transform(df_na).columns.tolist()
