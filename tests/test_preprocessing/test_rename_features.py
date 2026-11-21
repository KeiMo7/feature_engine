import pandas as pd
import pytest

from feature_engine.preprocessing import RenameFeatures


@pytest.fixture
def df():
    return pd.DataFrame({"x1": [1, 2, 3, 4], "x2": [5, 6, 7, 8], "x3": [9, 10, 11, 12]})


def test_rename_basic(df):
    transformer = RenameFeatures(rename_dict={"x1": "age", "x2": "height"})
    result = transformer.fit_transform(df)

    assert result.columns.tolist() == ["age", "height", "x3"]
    # values are untouched, only labels change
    pd.testing.assert_series_equal(result["age"], df["x1"], check_names=False)
    assert transformer.feature_renames_ == {"x1": "age", "x2": "height"}


def test_get_feature_names_out(df):
    transformer = RenameFeatures(rename_dict={"x1": "age"}).fit(df)
    assert transformer.get_feature_names_out() == ["age", "x2", "x3"]
    assert transformer.get_feature_names_out(["x1", "x2", "x3"]) == ["age", "x2", "x3"]
    with pytest.raises(ValueError):
        transformer.get_feature_names_out(["wrong", "x2", "x3"])


def test_non_dict_input_raises():
    with pytest.raises(ValueError):
        RenameFeatures(rename_dict=["not", "a", "dict"])
    with pytest.raises(ValueError):
        RenameFeatures(rename_dict={})


def test_unknown_column_raises(df):
    transformer = RenameFeatures(rename_dict={"missing": "new"})
    with pytest.raises(ValueError):
        transformer.fit(df)


def test_duplicate_target_raises(df):
    transformer = RenameFeatures(rename_dict={"x1": "z", "x2": "z"})
    with pytest.raises(ValueError):
        transformer.fit(df)


def test_target_already_exists_raises(df):
    transformer = RenameFeatures(rename_dict={"x1": "x2"})
    with pytest.raises(ValueError):
        transformer.fit(df)


def test_rename_preserves_dtypes(df):
    transformer = RenameFeatures(rename_dict={"x1": "age"}).fit(df)
    result = transformer.transform(df)
    assert result["age"].dtype == df["x1"].dtype


def test_integer_column_names():
    df = pd.DataFrame({0: [1, 2], 1: [3, 4]})
    transformer = RenameFeatures(rename_dict={0: 10}).fit(df)
    assert transformer.transform(df).columns.tolist() == [10, 1]


def test_existing_transform_results_unchanged(df):
    # the transformer must not alter the input dataframe
    original = df.copy()
    RenameFeatures(rename_dict={"x1": "age"}).fit_transform(df)
    pd.testing.assert_frame_equal(df, original)
