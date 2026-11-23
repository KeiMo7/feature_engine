import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from feature_engine.discretisation import MonotonicDiscretiser


@pytest.fixture(scope="module")
def df_monotonic():
    """Business-like data: bad rate increases monotonically with the score."""
    rng = np.random.default_rng(42)
    n = 2000
    score = rng.normal(0, 1, n)
    p_bad = 1 / (1 + np.exp(-(-1 + 2 * score)))
    y = pd.Series(rng.binomial(1, p_bad), name="bad")
    X = pd.DataFrame({"score": score, "noise": rng.normal(0, 1, n)})
    return X, y


@pytest.fixture(scope="module")
def df_with_missing(df_monotonic):
    X, y = df_monotonic
    X = X.copy()
    rng = np.random.default_rng(7)
    X.loc[rng.choice(X.index, 100, replace=False), "score"] = np.nan
    return X, y


def _interval_rows(table):
    return table[table["bin"] != "Missing"]


# ---------------------------------------------------------------------------
# monotonic constraint
# ---------------------------------------------------------------------------


def test_bad_rate_monotonic_increasing_and_min_bin_size(df_monotonic):
    X, y = df_monotonic
    md = MonotonicDiscretiser(
        variables=["score"], max_bins=5, min_bin_size=0.05, monotonic="increasing"
    )
    md.fit(X, y)

    table = _interval_rows(md.binning_table_["score"])
    assert table["bad_rate"].is_monotonic_increasing
    assert (table["count"] >= 0.05 * len(X)).all()
    assert len(table) <= 5


def test_bad_rate_monotonic_decreasing(df_monotonic):
    X, y = df_monotonic
    md = MonotonicDiscretiser(
        variables=["score"], max_bins=5, monotonic="decreasing"
    ).fit(X, y)

    table = _interval_rows(md.binning_table_["score"])
    assert table["bad_rate"].is_monotonic_decreasing
    assert md.monotonic_direction_["score"] == "decreasing"


def test_monotonic_auto_picks_a_valid_direction(df_monotonic):
    X, y = df_monotonic
    md = MonotonicDiscretiser(variables=["score"], monotonic="auto").fit(X, y)

    direction = md.monotonic_direction_["score"]
    assert direction in ("increasing", "decreasing")
    table = _interval_rows(md.binning_table_["score"])
    if direction == "increasing":
        assert table["bad_rate"].is_monotonic_increasing
    else:
        assert table["bad_rate"].is_monotonic_decreasing


def test_conflicting_direction_reports_error_not_silent_flip(df_monotonic):
    # asking for a decreasing bad rate on increasing data must not silently
    # return increasing bins: the result is decreasing or an error is raised
    X, y = df_monotonic
    md = MonotonicDiscretiser(variables=["score"], monotonic="decreasing")
    md.fit(X, y)
    table = _interval_rows(md.binning_table_["score"])
    assert table["bad_rate"].is_monotonic_decreasing


def test_monotonic_none_degrades_to_plain_reproducible_binning(df_monotonic):
    X, y = df_monotonic
    params = dict(variables=["score"], monotonic=None, max_bins=4)
    md1 = MonotonicDiscretiser(**params).fit(X, y)
    md2 = MonotonicDiscretiser(**params).fit(X, y)
    assert md1.binner_dict_ == md2.binner_dict_
    assert md1.monotonic_direction_["score"] is None
    # no monotonic guarantee, but constraints still hold
    table = _interval_rows(md1.binning_table_["score"])
    assert len(table) <= 4


def test_zigzag_rates_are_merged_into_closest_monotone_solution():
    # alternating bad rates: a tree-like split would jump up and down
    rng = np.random.default_rng(3)
    n = 4000
    x = np.arange(n, dtype=float)
    p = np.where((x // 100) % 2 == 0, 0.1, 0.3) + x / n * 0.4
    y = pd.Series(rng.binomial(1, p))
    X = pd.DataFrame({"x": x})

    md = MonotonicDiscretiser(
        max_bins=6, min_bin_size=0.1, monotonic="increasing"
    ).fit(X, y)
    table = _interval_rows(md.binning_table_["x"])

    # the feasible solution: monotone bad rates and min bin size honoured
    assert table["bad_rate"].is_monotonic_increasing
    assert (table["count"] >= 0.1 * n).all()
    assert len(table) <= 6


# ---------------------------------------------------------------------------
# missing values
# ---------------------------------------------------------------------------


def test_missing_values_form_their_own_bin(df_with_missing):
    X, y = df_with_missing
    md = MonotonicDiscretiser(variables=["score"], max_bins=5).fit(X, y)

    Xt = md.transform(X)
    assert (Xt.loc[X["score"].isna(), "score"] == -1).all()
    # no normal bin contains missing values
    assert Xt["score"].notna().all()

    table = md.binning_table_["score"]
    missing_row = table[table["bin"] == "Missing"]
    assert len(missing_row) == 1
    assert missing_row["count"].iloc[0] == 100
    assert missing_row["bad"].iloc[0] == y[X["score"].isna()].sum()
    # the missing bin participates in the statistics
    assert np.isfinite(missing_row["woe"].iloc[0])
    assert table["count"].sum() == len(X)


def test_missing_bin_false_returns_nan(df_with_missing):
    X, y = df_with_missing
    md = MonotonicDiscretiser(variables=["score"], missing_bin=False).fit(X, y)
    Xt = md.transform(X)
    assert Xt.loc[X["score"].isna(), "score"].isna().all()


# ---------------------------------------------------------------------------
# binning table output
# ---------------------------------------------------------------------------


def test_binning_table_content(df_monotonic):
    X, y = df_monotonic
    md = MonotonicDiscretiser(variables=["score"], max_bins=5).fit(X, y)
    table = md.binning_table_["score"]

    expected_cols = {
        "bin", "lower", "upper", "count", "bad", "good",
        "bad_rate", "woe", "iv", "is_boundary",
    }
    assert expected_cols.issubset(table.columns)

    intervals = _interval_rows(table)
    # bounds are ordered and match binner_dict_
    edges = md.binner_dict_["score"]
    assert intervals["lower"].tolist() == edges[:-1]
    assert intervals["upper"].tolist() == edges[1:]
    # counts add up
    assert intervals["count"].sum() == len(X)
    assert (intervals["bad"] + intervals["good"] == intervals["count"]).all()
    # boundary bins are flagged
    assert intervals["is_boundary"].iloc[0]
    assert intervals["is_boundary"].iloc[-1]
    assert not intervals["is_boundary"].iloc[1:-1].any()
    # iv attribute matches the table
    assert md.iv_["score"] == pytest.approx(table["iv"].sum())
    # no raw identifiers or row-level data leak into the output
    assert set(table.columns) == expected_cols


def test_woe_and_iv_are_finite(df_monotonic):
    X, y = df_monotonic
    md = MonotonicDiscretiser(variables=["score"]).fit(X, y)
    table = md.binning_table_["score"]
    assert np.isfinite(table["woe"]).all()
    assert np.isfinite(table["iv"]).all()


# ---------------------------------------------------------------------------
# out-of-range values
# ---------------------------------------------------------------------------


def test_out_of_range_values_fall_into_boundary_bins(df_monotonic):
    X, y = df_monotonic
    md = MonotonicDiscretiser(variables=["score"], max_bins=5).fit(X, y)

    X_new = pd.DataFrame(
        {"score": [-1e6, 1e6], "noise": [0.0, 0.0]}
    )
    Xt = md.transform(X_new)
    n_bins = len(md.binner_dict_["score"]) - 1
    assert Xt["score"].tolist() == [0, n_bins - 1]


# ---------------------------------------------------------------------------
# determinism and frozen boundaries
# ---------------------------------------------------------------------------


def test_repeated_fit_gives_identical_boundaries(df_with_missing):
    X, y = df_with_missing
    md1 = MonotonicDiscretiser(variables=["score"], max_bins=5).fit(X, y)
    md2 = MonotonicDiscretiser(variables=["score"], max_bins=5).fit(X, y)

    assert md1.binner_dict_ == md2.binner_dict_
    for var in md1.binning_table_:
        pd.testing.assert_frame_equal(
            md1.binning_table_[var], md2.binning_table_[var]
        )


def test_transform_does_not_use_y_and_boundaries_are_frozen(df_monotonic):
    X, y = df_monotonic
    md = MonotonicDiscretiser(variables=["score"], max_bins=5).fit(X, y)
    boundaries_after_fit = dict(md.binner_dict_)

    # transform data with a very different distribution: bins must not move
    X_new = pd.DataFrame({"score": np.linspace(-3, 3, 50), "noise": 0.0})
    Xt1 = md.transform(X_new)
    assert md.binner_dict_ == boundaries_after_fit

    # same value always lands in the same bin
    Xt2 = md.transform(X_new)
    pd.testing.assert_frame_equal(Xt1, Xt2)

    # transform has no y parameter
    import inspect

    assert "y" not in inspect.signature(md.transform).parameters


# ---------------------------------------------------------------------------
# special variables
# ---------------------------------------------------------------------------


def test_constant_variable_gets_a_single_bin():
    X = pd.DataFrame({"const": [5.0] * 100})
    y = pd.Series([0, 1] * 50)
    md = MonotonicDiscretiser().fit(X, y)

    assert md.binner_dict_["const"] == [float("-inf"), float("inf")]
    table = _interval_rows(md.binning_table_["const"])
    assert len(table) == 1
    assert table["count"].iloc[0] == 100
    assert (md.transform(X)["const"] == 0).all()


def test_variable_with_too_few_non_missing_values_raises():
    X = pd.DataFrame({"sparse": [1.0, 2.0, 3.0] + [np.nan] * 197})
    y = pd.Series([0, 1] * 100)
    md = MonotonicDiscretiser(min_bin_size=10)
    with pytest.raises(ValueError, match="sparse"):
        md.fit(X, y)


def test_insufficient_samples_error_states_shortfall_and_suggestion():
    X = pd.DataFrame({"var": np.arange(50.0)})
    y = pd.Series([0, 1] * 25)
    md = MonotonicDiscretiser(min_bin_size=100)
    with pytest.raises(ValueError) as err:
        md.fit(X, y)
    msg = str(err.value)
    assert "50" in msg and "100" in msg  # actual vs required
    assert "shortfall: 50" in msg
    assert "min_bin_size" in msg  # suggested configuration


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------


def test_inf_values_are_rejected_locating_the_column():
    X = pd.DataFrame({"ok": [1.0, 2.0, 3.0], "bad_col": [1.0, np.inf, 3.0]})
    y = pd.Series([0, 1, 0])
    with pytest.raises(ValueError, match="bad_col"):
        MonotonicDiscretiser().fit(X, y)


def test_inf_values_rejected_at_transform(df_monotonic):
    X, y = df_monotonic
    md = MonotonicDiscretiser(variables=["score"]).fit(X, y)
    X_new = pd.DataFrame({"score": [np.inf], "noise": [1.0]})
    with pytest.raises(ValueError, match="score"):
        md.transform(X_new)


def test_non_binary_target_is_rejected(df_monotonic):
    X, _ = df_monotonic
    with pytest.raises(ValueError, match="binary target"):
        MonotonicDiscretiser().fit(X, pd.Series(np.resize([0, 1, 2], len(X))))


def test_single_class_target_is_rejected(df_monotonic):
    X, _ = df_monotonic
    with pytest.raises(ValueError, match="both classes"):
        MonotonicDiscretiser().fit(X, pd.Series([0] * len(X)))


def test_fit_requires_y(df_monotonic):
    X, _ = df_monotonic
    with pytest.raises(TypeError):
        MonotonicDiscretiser().fit(X)


@pytest.mark.parametrize(
    "params",
    [
        {"max_bins": 0},
        {"max_bins": 2.5},
        {"min_bin_size": 0},
        {"min_bin_size": 1.5},
        {"min_bin_size": "10"},
        {"monotonic": "up"},
        {"initial_bins": 1},
        {"bin_output": "score"},
        {"missing_bin": "yes"},
        {"precision": 0},
    ],
)
def test_init_param_validation(params):
    with pytest.raises(ValueError):
        MonotonicDiscretiser(**params)


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------


def test_bin_output_woe(df_monotonic):
    X, y = df_monotonic
    md = MonotonicDiscretiser(variables=["score"], bin_output="woe").fit(X, y)
    Xt = md.transform(X)
    table = md.binning_table_["score"]
    assert Xt["score"].isin(table["woe"]).all()


def test_bin_output_boundaries(df_with_missing):
    X, y = df_with_missing
    md = MonotonicDiscretiser(
        variables=["score"], bin_output="boundaries"
    ).fit(X, y)
    Xt = md.transform(X)
    assert (Xt.loc[X["score"].isna(), "score"] == "Missing").all()
    assert Xt.loc[X["score"].notna(), "score"].str.startswith("(").all()


def test_works_within_pipeline(df_monotonic):
    X, y = df_monotonic
    pipe = Pipeline(
        [("bin", MonotonicDiscretiser(variables=["score"], max_bins=4))]
    )
    Xt = pipe.fit_transform(X, y)
    assert Xt["score"].nunique() <= 5  # 4 interval bins + missing
    assert list(Xt.columns) == list(X.columns)


def test_get_feature_names_out(df_monotonic):
    X, y = df_monotonic
    md = MonotonicDiscretiser(variables=["score"]).fit(X, y)
    assert md.get_feature_names_out() == list(X.columns)
