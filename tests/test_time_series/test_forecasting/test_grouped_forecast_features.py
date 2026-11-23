import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from feature_engine.timeseries.forecasting import (
    GroupedExpandingWindowFeatures,
    GroupedLagFeatures,
    GroupedWindowFeatures,
)


@pytest.fixture
def three_entities():
    """Three business lines with irregular timestamps and gaps.

    - A is sampled daily and then skips 2022-01-04 and 2022-01-05.
    - B is sampled every other day.
    - C is sampled irregularly.
    """
    return pd.DataFrame(
        {
            "store": ["A", "A", "A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "ts": pd.to_datetime(
                [
                    "2022-01-01",
                    "2022-01-02",
                    "2022-01-03",
                    "2022-01-06",
                    "2022-01-07",
                    "2022-01-01",
                    "2022-01-03",
                    "2022-01-05",
                    "2022-01-02",
                    "2022-01-04",
                    "2022-01-09",
                ]
            ),
            "sales": [1.0, 2.0, 3.0, 6.0, 7.0, 10.0, 30.0, 50.0, 100.0, 200.0, 700.0],
            "qty": [10.0, 20.0, 30.0, 60.0, 70.0, 1.0, 3.0, 5.0, 2.0, 4.0, 9.0],
        }
    )


def _assert_series_equal(actual, expected):
    actual = pd.Series(actual).reset_index(drop=True)
    expected = pd.Series(expected, dtype=float).reset_index(drop=True)
    assert_frame_equal(
        actual.to_frame("value"), expected.to_frame("value"), check_dtype=False
    )


def _manual_window(frame, window, value="sales", function="mean", horizon="0D"):
    """Reference: real-time interval ``(cutoff - window, cutoff]`` per entity.

    The interval is ``[timestamp - window, timestamp)`` for a zero horizon and
    ``[timestamp - horizon - window, timestamp - horizon)`` otherwise; the
    current row is always excluded.
    """
    out = pd.Series(np.nan, index=range(len(frame)), dtype=float)
    delta = pd.Timedelta(window)
    span = pd.Timedelta(horizon)
    for _, block in frame.groupby("store", sort=False):
        block = block.sort_values("ts", kind="mergesort")
        times = pd.DatetimeIndex(block["ts"])
        values = block[value].to_numpy()
        for offset, (position, timestamp) in enumerate(zip(block.index, times)):
            cutoff = timestamp - span
            end = times.searchsorted(cutoff, side="right")
            if timestamp <= cutoff:
                end = times.searchsorted(timestamp, side="left")
            start = times.searchsorted(cutoff - delta, side="left")
            segment = values[start:end]
            if len(segment):
                out.loc[position] = getattr(np, function)(segment)
    return out


@pytest.mark.parametrize("function", ["mean", "sum", "max"])
def test_time_windows_match_manual_computation(three_entities, function):
    frame = three_entities
    tr = GroupedWindowFeatures(
        entity_variables="store",
        timestamp="ts",
        window="2D",
        functions=function,
    )
    out = tr.fit_transform(frame)

    expected = _manual_window(frame, "2D", function=function)
    _assert_series_equal(
        out["sales_window_2D_" + function],
        expected,
    )


def test_windows_use_real_time_not_row_position(three_entities):
    frame = three_entities
    tr = GroupedWindowFeatures(
        entity_variables="store",
        timestamp="ts",
        window="1D",
        functions="sum",
    )
    out = tr.fit_transform(frame)

    # Entity A, 2022-01-03: the previous observation is 2022-01-02, exactly one
    # day old. The inclusive lower bound [01-02, 01-03) includes it.
    row = out[(out["store"] == "A") & (out["ts"] == pd.Timestamp("2022-01-03"))]
    assert row["sales_window_1D_sum"].iloc[0] == 2.0

    # Entity C, 2022-01-04: the previous observation is 2022-01-02 (two days
    # old), while a one-day window reaches only to 2022-01-03, so it is empty.
    # A row-based shift would wrongly use 2022-01-02.
    row = out[(out["store"] == "C") & (out["ts"] == pd.Timestamp("2022-01-04"))]
    assert np.isnan(row["sales_window_1D_sum"].iloc[0])

    tr = GroupedWindowFeatures(
        entity_variables="store",
        timestamp="ts",
        window="2D",
        functions="sum",
    )
    out = tr.fit_transform(frame)

    # Entity A, 2022-01-03: the two-day interval [01-01, 01-03) contains
    # 2022-01-01 and 2022-01-02.
    row = out[(out["store"] == "A") & (out["ts"] == pd.Timestamp("2022-01-03"))]
    assert row["sales_window_2D_sum"].iloc[0] == 3.0

    # Entity A, 2022-01-06: the two-day interval [01-04, 01-06) is empty,
    # although the previous row is 2022-01-03.
    row = out[(out["store"] == "A") & (out["ts"] == pd.Timestamp("2022-01-06"))]
    assert np.isnan(row["sales_window_2D_sum"].iloc[0])

    # Entity C, 2022-01-09: the two-day interval [01-07, 01-09) is empty, so
    # 2022-01-04 is not pulled in even though it is the previous row.
    row = out[(out["store"] == "C") & (out["ts"] == pd.Timestamp("2022-01-09"))]
    assert np.isnan(row["sales_window_2D_sum"].iloc[0])


def test_no_cross_entity_leakage(three_entities):
    frame = three_entities
    tr = GroupedWindowFeatures(
        entity_variables="store",
        timestamp="ts",
        window=2,
        functions="sum",
    )
    out = tr.fit_transform(frame)

    for entity in ("A", "B", "C"):
        block = out[out["store"] == entity].sort_values("ts")
        # The first row of each entity has no history, even if a previous
        # entity ends right before it in the table.
        assert np.isnan(block["sales_window_2_sum"].iloc[0])

    # Entity C's second row has a single prior observation, so a full
    # two-observation window is not available yet; the third row combines the
    # second and third C values only.
    block = out[out["store"] == "C"].sort_values("ts")
    assert np.isnan(block["sales_window_2_sum"].iloc[1])
    assert block["sales_window_2_sum"].iloc[2] == 300.0

    # Entity B's third row contains B values only: 10 + 30.
    block = out[out["store"] == "B"].sort_values("ts")
    assert block["sales_window_2_sum"].iloc[2] == 40.0


def test_horizon_excludes_recent_and_future_rows(three_entities):
    frame = three_entities
    tr = GroupedWindowFeatures(
        entity_variables="store",
        timestamp="ts",
        window="3D",
        functions="sum",
        horizon="1D",
    )
    out = tr.fit_transform(frame)

    # Entity A, 2022-01-06: a 1D horizon moves the cutoff to 2022-01-05, and
    # the 3D window spans [2022-01-02, 2022-01-05), so 2022-01-02 and
    # 2022-01-03 count.
    row = out[(out["store"] == "A") & (out["ts"] == pd.Timestamp("2022-01-06"))]
    assert row["sales_window_3D_sum"].iloc[0] == 5.0

    # Entity A, 2022-01-02: cutoff 2022-01-01, only that row qualifies.
    row = out[(out["store"] == "A") & (out["ts"] == pd.Timestamp("2022-01-02"))]
    assert row["sales_window_3D_sum"].iloc[0] == 1.0


def test_horizon_never_sees_future(three_entities):
    frame = three_entities
    params = dict(entity_variables="store", timestamp="ts", window="2D", horizon="1D")
    tr = GroupedWindowFeatures(**params)
    out = tr.fit_transform(frame)

    perturbed = frame.copy()
    mask = perturbed["ts"] >= pd.Timestamp("2022-01-06")
    perturbed.loc[mask, "sales"] = perturbed.loc[mask, "sales"] * 1000

    tr2 = GroupedWindowFeatures(**params)
    out2 = tr2.fit_transform(perturbed)

    early = frame["ts"] < pd.Timestamp("2022-01-06")
    assert_frame_equal(
        out.loc[early, ["sales_window_2D_mean"]],
        out2.loc[early, ["sales_window_2D_mean"]],
    )


def test_horizon_in_rows_uses_previous_observations(three_entities):
    frame = three_entities
    tr = GroupedExpandingWindowFeatures(
        entity_variables="store", timestamp="ts", horizon=2
    )
    out = tr.fit_transform(frame)

    # Entity A: 2022-01-03 (third observation) can only use the first one.
    row = out[(out["store"] == "A") & (out["ts"] == pd.Timestamp("2022-01-03"))]
    assert row["sales_expanding_mean"].iloc[0] == 1.0

    # Entity A: the fourth observation uses the first two.
    row = out[(out["store"] == "A") & (out["ts"] == pd.Timestamp("2022-01-06"))]
    assert row["sales_expanding_mean"].iloc[0] == 1.5


def test_lags_with_periods(three_entities):
    frame = three_entities
    tr = GroupedLagFeatures(entity_variables="store", timestamp="ts", periods=[1, 2])
    out = tr.fit_transform(frame)

    block = out[out["store"] == "A"].sort_values("ts")
    _assert_series_equal(block["sales_lag_1"], [np.nan, 1.0, 2.0, 3.0, 6.0])
    _assert_series_equal(block["sales_lag_2"], [np.nan, np.nan, 1.0, 2.0, 3.0])

    # Entity B's first lag is its own NaN, not A's last value.
    block = out[out["store"] == "B"].sort_values("ts")
    _assert_series_equal(block["sales_lag_1"], [np.nan, 10.0, 30.0])


def test_lags_with_freq_respect_gaps(three_entities):
    frame = three_entities
    tr = GroupedLagFeatures(entity_variables="store", timestamp="ts", freq="2D")
    out = tr.fit_transform(frame)

    # A 2022-01-03: latest observation at or before 2022-01-01 is 01-01.
    row = out[(out["store"] == "A") & (out["ts"] == pd.Timestamp("2022-01-03"))]
    assert row["sales_lag_2D"].iloc[0] == 1.0

    # A 2022-01-06: latest at or before 2022-01-04 is 01-03.
    row = out[(out["store"] == "A") & (out["ts"] == pd.Timestamp("2022-01-06"))]
    assert row["sales_lag_2D"].iloc[0] == 3.0

    # A 2022-01-02: nothing at or before 2021-12-31.
    row = out[(out["store"] == "A") & (out["ts"] == pd.Timestamp("2022-01-02"))]
    assert np.isnan(row["sales_lag_2D"].iloc[0])

    # C 2022-01-09: latest at or before 2022-01-07 is 01-04.
    row = out[(out["store"] == "C") & (out["ts"] == pd.Timestamp("2022-01-09"))]
    assert row["sales_lag_2D"].iloc[0] == 200.0


def test_different_sampling_frequencies_are_not_resampled():
    frame = pd.DataFrame(
        {
            "store": ["daily"] * 4 + ["weekly"] * 3,
            "ts": pd.to_datetime(
                [
                    "2022-01-01",
                    "2022-01-02",
                    "2022-01-03",
                    "2022-01-04",
                    "2022-01-01",
                    "2022-01-08",
                    "2022-01-15",
                ]
            ),
            "sales": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0],
        }
    )
    tr = GroupedLagFeatures(entity_variables="store", timestamp="ts", periods=1)
    out = tr.fit_transform(frame)

    _assert_series_equal(
        out[out["store"] == "weekly"]["sales_lag_1"], [np.nan, 10.0, 20.0]
    )
    _assert_series_equal(
        out[out["store"] == "daily"]["sales_lag_1"], [np.nan, 1.0, 2.0, 3.0]
    )


def test_original_row_order_and_columns_are_preserved(three_entities):
    frame = three_entities.sample(frac=1, random_state=0)
    original = frame.copy()
    tr = GroupedWindowFeatures(
        entity_variables="store", timestamp="ts", window=2, functions="mean"
    )
    out = tr.fit_transform(frame)

    assert out.index.equals(original.index)
    assert_frame_equal(
        out[["store", "ts", "sales", "qty"]],
        original[["store", "ts", "sales", "qty"]],
    )
    for position in original.index:
        assert out.loc[position, "store"] == original.loc[position, "store"]
        assert out.loc[position, "ts"] == original.loc[position, "ts"]


def test_unsorted_input_is_sorted_per_entity(three_entities):
    shuffled = three_entities.sample(frac=1, random_state=42)
    tr = GroupedLagFeatures(entity_variables="store", timestamp="ts", periods=1)
    out = tr.fit_transform(shuffled)

    for _, block in shuffled.groupby("store"):
        block = block.sort_values("ts", kind="mergesort")
        expected = [np.nan] + block["sales"].iloc[:-1].tolist()
        got = out.loc[block.index, "sales_lag_1"].tolist()
        _assert_series_equal(got, expected)


def test_min_periods_controls_insufficient_history(three_entities):
    frame = three_entities
    tr = GroupedWindowFeatures(
        entity_variables="store",
        timestamp="ts",
        window=3,
        min_periods=2,
        functions="count",
    )
    out = tr.fit_transform(frame)

    # A's second row has 1 prior observation: below min_periods, so NaN.
    row = out[(out["store"] == "A") & (out["ts"] == pd.Timestamp("2022-01-02"))]
    assert np.isnan(row["sales_window_3_count"].iloc[0])

    # A's third row has 2 prior observations: value emitted.
    row = out[(out["store"] == "A") & (out["ts"] == pd.Timestamp("2022-01-03"))]
    assert row["sales_window_3_count"].iloc[0] == 2.0

    # With min_periods=1 the same row would emit a value.
    tr_small = GroupedLagFeatures(entity_variables="store", timestamp="ts", periods=1)
    assert tr_small.fit_transform(frame).loc[1, "sales_lag_1"] == 1.0


def test_multiple_variables_windows_and_functions_names(three_entities):
    frame = three_entities
    tr = GroupedWindowFeatures(
        entity_variables="store",
        timestamp="ts",
        variables=["sales", "qty"],
        window=[2, "3D"],
        functions=["mean", "sum"],
    )
    tr.fit(frame)
    names = tr.get_feature_names_out()

    assert names == [
        "store",
        "ts",
        "sales",
        "qty",
        "sales_window_2_mean",
        "sales_window_2_sum",
        "qty_window_2_mean",
        "qty_window_2_sum",
        "sales_window_3D_mean",
        "sales_window_3D_sum",
        "qty_window_3D_mean",
        "qty_window_3D_sum",
    ]
    assert list(tr.transform(frame).columns) == names


def test_duplicate_timestamps_conflict_is_reported():
    frame = pd.DataFrame(
        {
            "store": ["A", "A", "A"],
            "ts": pd.to_datetime(["2022-01-01", "2022-01-01", "2022-01-02"]),
            "sales": [1.0, 2.0, 3.0],
        }
    )
    tr = GroupedLagFeatures(entity_variables="store", timestamp="ts")
    with pytest.raises(ValueError, match="duplicated"):
        tr.fit(frame)


def test_duplicate_timestamps_can_be_kept_deterministically():
    frame = pd.DataFrame(
        {
            "store": ["A", "A", "A", "A"],
            "ts": pd.to_datetime(
                ["2022-01-01", "2022-01-02", "2022-01-02", "2022-01-03"]
            ),
            "sales": [1.0, 2.0, 4.0, 8.0],
        }
    )
    tr = GroupedLagFeatures(
        entity_variables="store", timestamp="ts", periods=1, duplicates="keep"
    )
    out = tr.fit_transform(frame)

    # Rows on the same timestamp do not enter each other's history, but the
    # rows before the shared timestamp still do.
    assert out.loc[1, "sales_lag_1"] == 1.0
    assert out.loc[2, "sales_lag_1"] == 1.0
    assert out.loc[3, "sales_lag_1"] == 4.0


def test_duplicate_timestamps_first_and_last():
    frame = pd.DataFrame(
        {
            "store": ["A", "A", "A"],
            "ts": pd.to_datetime(["2022-01-01", "2022-01-01", "2022-01-02"]),
            "sales": [1.0, 2.0, 3.0],
        }
    )
    first = GroupedLagFeatures(
        entity_variables="store", timestamp="ts", duplicates="first"
    )
    out_first = first.fit_transform(frame)
    assert out_first["sales"].tolist() == [1.0, 3.0]

    last = GroupedLagFeatures(
        entity_variables="store", timestamp="ts", duplicates="last"
    )
    out_last = last.fit_transform(frame)
    assert out_last["sales"].tolist() == [2.0, 3.0]


def test_unseen_entity_requires_explicit_cold_start():
    frame = pd.DataFrame(
        {
            "store": ["A", "A", "B"],
            "ts": pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-02"]),
            "sales": [1.0, 2.0, 3.0],
        }
    )
    tr = GroupedWindowFeatures(entity_variables="store", timestamp="ts", window=1)
    tr.fit(frame)

    new_entity = frame.copy()
    new_entity.loc[2, "store"] = "C"
    with pytest.raises(ValueError, match="not seen during fit"):
        tr.transform(new_entity)

    nan_tr = GroupedWindowFeatures(
        entity_variables="store", timestamp="ts", window=1, cold_start="nan"
    )
    nan_tr.fit(frame)
    out = nan_tr.transform(new_entity)
    assert np.isnan(out.loc[2, "sales_window_1_mean"])

    fill_tr = GroupedWindowFeatures(
        entity_variables="store",
        timestamp="ts",
        window=1,
        cold_start="fill",
        cold_start_value=-1.0,
    )
    fill_tr.fit(frame)
    out = fill_tr.transform(new_entity)
    assert out.loc[2, "sales_window_1_mean"] == -1.0
    # known entities keep their real values
    assert out.loc[1, "sales_window_1_mean"] == 1.0


def test_cold_start_fill_requires_value():
    with pytest.raises(ValueError, match="cold_start_value"):
        GroupedWindowFeatures(
            entity_variables="store", timestamp="ts", cold_start="fill"
        )


def test_missing_entity_value_is_located():
    frame = pd.DataFrame(
        {
            "store": ["A", None, "B"],
            "ts": pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-03"]),
            "sales": [1.0, 2.0, 3.0],
        }
    )
    tr = GroupedLagFeatures(entity_variables="store", timestamp="ts")
    with pytest.raises(ValueError, match="entity column.*missing"):
        tr.fit(frame)


def test_unparseable_timestamp_is_located():
    frame = pd.DataFrame(
        {
            "store": ["A", "A", "A"],
            "ts": ["2022-01-01", "not-a-date", "2022-01-03"],
            "sales": [1.0, 2.0, 3.0],
        }
    )
    tr = GroupedLagFeatures(entity_variables="store", timestamp="ts")
    with pytest.raises(ValueError, match="unparseable timestamps"):
        tr.fit(frame)


def test_horizon_without_history_explains_reason():
    frame = pd.DataFrame(
        {
            "store": ["A", "A", "A"],
            "ts": pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-03"]),
            "sales": [1.0, 2.0, 3.0],
        }
    )
    tr = GroupedWindowFeatures(
        entity_variables="store", timestamp="ts", window="2D", horizon="30D"
    )
    with pytest.raises(ValueError, match="leaves no usable history"):
        tr.fit_transform(frame)


def test_drop_original_and_drop_na(three_entities):
    frame = three_entities
    tr = GroupedWindowFeatures(
        entity_variables="store",
        timestamp="ts",
        variables=["sales"],
        window=2,
        drop_original=True,
    )
    out = tr.fit_transform(frame)
    assert "sales" not in out.columns
    assert "qty" in out.columns

    tr_na = GroupedWindowFeatures(
        entity_variables="store", timestamp="ts", window=3, drop_na=True
    )
    out_na = tr_na.fit_transform(frame)
    assert not out_na["sales_window_3_mean"].isna().any()


def test_transform_x_y(three_entities):
    frame = three_entities
    y = pd.Series(np.arange(len(frame)), index=frame.index)
    tr = GroupedWindowFeatures(
        entity_variables="store", timestamp="ts", window=3, drop_na=True
    )
    tr.fit(frame)
    Xt, yt = tr.transform_x_y(frame, y)
    assert len(Xt) == len(yt)
    assert (Xt.index == yt.index).all()


def test_requires_entity_and_timestamp_columns():
    frame = pd.DataFrame({"store": ["A"], "ts": pd.to_datetime(["2022-01-01"])})
    with pytest.raises(ValueError, match="entity_variables"):
        GroupedLagFeatures(timestamp="ts").fit(frame)
    with pytest.raises(ValueError, match="timestamp"):
        GroupedLagFeatures(entity_variables="store").fit(frame)
    with pytest.raises(ValueError, match="not present"):
        GroupedLagFeatures(entity_variables="missing", timestamp="ts").fit(frame)


def test_variables_exclude_entity_and_timestamp(three_entities):
    frame = three_entities
    tr = GroupedWindowFeatures(entity_variables="store", timestamp="ts", window=2)
    tr.fit(frame)
    assert "store" not in tr.variables_
    assert "ts" not in tr.variables_

    with pytest.raises((TypeError, ValueError)):
        GroupedWindowFeatures(
            entity_variables="store",
            timestamp="ts",
            variables=["store"],
        ).fit(frame)


def test_invalid_parameters():
    with pytest.raises(ValueError):
        GroupedLagFeatures(entity_variables="s", timestamp="t", periods=0)
    with pytest.raises(ValueError):
        GroupedLagFeatures(entity_variables="s", timestamp="t", periods=[1, 1])
    with pytest.raises(ValueError):
        GroupedLagFeatures(entity_variables="s", timestamp="t", freq="1M")
    with pytest.raises(ValueError):
        GroupedLagFeatures(entity_variables="s", timestamp="t", horizon=-1)
    with pytest.raises(ValueError):
        GroupedLagFeatures(entity_variables="s", timestamp="t", duplicates="pizza")
    with pytest.raises(ValueError):
        GroupedLagFeatures(entity_variables="s", timestamp="t", cold_start="pizza")
    with pytest.raises(ValueError):
        GroupedWindowFeatures(entity_variables="s", timestamp="t", window=0)
    with pytest.raises(ValueError):
        GroupedWindowFeatures(entity_variables="s", timestamp="t", window=[2, 2])
    with pytest.raises(ValueError):
        GroupedWindowFeatures(entity_variables="s", timestamp="t", functions=1)
    with pytest.raises(ValueError):
        GroupedWindowFeatures(
            entity_variables="s", timestamp="t", functions=["mean", "mean"]
        )
    with pytest.raises(ValueError):
        GroupedExpandingWindowFeatures(
            entity_variables="s", timestamp="t", min_periods=0
        )


def test_multiple_entity_columns():
    frame = pd.DataFrame(
        {
            "region": ["EU", "EU", "EU", "US", "US"],
            "store": ["A", "A", "A", "A", "A"],
            "ts": pd.to_datetime(
                ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-01", "2022-01-02"]
            ),
            "sales": [1.0, 2.0, 3.0, 10.0, 20.0],
        }
    )
    tr = GroupedLagFeatures(
        entity_variables=["region", "store"], timestamp="ts", periods=1
    )
    out = tr.fit_transform(frame)
    _assert_series_equal(out["sales_lag_1"], [np.nan, 1.0, 2.0, np.nan, 10.0])


def test_timestamp_timezone_is_supported():
    frame = pd.DataFrame(
        {
            "store": ["A", "A", "A"],
            "ts": pd.to_datetime(
                ["2022-01-01", "2022-01-02", "2022-01-04"]
            ).tz_localize("UTC"),
            "sales": [1.0, 2.0, 4.0],
        }
    )
    tr = GroupedWindowFeatures(
        entity_variables="store",
        timestamp="ts",
        window="2D",
        functions="sum",
    )
    out = tr.fit_transform(frame)
    # 2022-01-04: the two-day interval [01-02, 01-04) includes the observation
    # exactly two days old.
    _assert_series_equal(out["sales_window_2D_sum"], [np.nan, 1.0, 2.0])


def test_expanding_no_leakage_across_entities(three_entities):
    frame = three_entities
    tr = GroupedExpandingWindowFeatures(
        entity_variables="store", timestamp="ts", functions=["mean", "count"]
    )
    out = tr.fit_transform(frame)

    block = out[out["store"] == "A"].sort_values("ts")
    counts = block["sales_expanding_count"].tolist()
    assert np.isnan(counts[0])
    assert counts[1] == 1.0
    assert counts[-1] == 4.0

    for entity in ("A", "B", "C"):
        first = out[out["store"] == entity].sort_values("ts").iloc[0]
        assert np.isnan(first["sales_expanding_mean"])


def test_no_new_dependencies_are_required():
    import feature_engine.timeseries.forecasting as forecasting

    assert hasattr(forecasting, "GroupedLagFeatures")
    assert hasattr(forecasting, "GroupedWindowFeatures")
    assert hasattr(forecasting, "GroupedExpandingWindowFeatures")


def test_original_ungrouped_classes_are_unchanged():
    from feature_engine.timeseries.forecasting import (
        ExpandingWindowFeatures,
        LagFeatures,
        WindowFeatures,
    )

    assert LagFeatures.__module__.endswith("lag_features")
    assert WindowFeatures.__module__.endswith("window_features")
    assert ExpandingWindowFeatures.__module__.endswith("expanding_window_features")


def test_grouped_matches_ungrouped_for_single_entity():
    from feature_engine.timeseries.forecasting import (
        ExpandingWindowFeatures,
        LagFeatures,
        WindowFeatures,
    )

    index = pd.date_range("2020-01-01", periods=12, freq="D")
    frame = pd.DataFrame(
        {"x": np.arange(1.0, 13.0), "y": np.arange(2.0, 26.0, 2.0)}, index=index
    )
    grouped_frame = frame.reset_index().rename(columns={"index": "ts"})
    grouped_frame["entity"] = "A"

    cases = [
        (
            LagFeatures(periods=[1, 2]),
            GroupedLagFeatures(
                entity_variables="entity", timestamp="ts", periods=[1, 2]
            ),
            "lag",
        ),
        (
            WindowFeatures(window=[2, 3], functions=["mean", "sum"]),
            GroupedWindowFeatures(
                entity_variables="entity",
                timestamp="ts",
                window=[2, 3],
                functions=["mean", "sum"],
            ),
            "window",
        ),
        (
            ExpandingWindowFeatures(functions=["mean", "sum"]),
            GroupedExpandingWindowFeatures(
                entity_variables="entity", timestamp="ts", functions=["mean", "sum"]
            ),
            "expanding",
        ),
    ]

    for old, new, pattern in cases:
        old_out = old.fit_transform(frame)
        new_out = new.fit_transform(grouped_frame)
        old_features = old_out.filter(like=pattern).to_numpy()
        new_features = new_out.filter(like=pattern).to_numpy()
        np.testing.assert_allclose(old_features, new_features, equal_nan=True)


def test_sklearn_get_params_and_clone():
    from sklearn.base import clone

    tr = GroupedWindowFeatures(
        entity_variables=["region", "store"],
        timestamp="ts",
        variables=["sales"],
        window=[2, "3D"],
        functions=["mean", "sum"],
        horizon="1D",
        cold_start="nan",
        duplicates="keep",
    )
    params = tr.get_params()
    assert params["entity_variables"] == ["region", "store"]
    assert params["window"] == [2, "3D"]
    assert params["horizon"] == "1D"

    cloned = clone(tr)
    assert cloned.get_params() == params


def test_sklearn_pipeline_with_pandas_output(three_entities):
    from sklearn.pipeline import Pipeline

    frame = three_entities
    tr = GroupedWindowFeatures(
        entity_variables="store", timestamp="ts", window=2, functions="mean"
    )
    pipe = Pipeline([("features", tr)]).set_output(transform="pandas")

    direct = tr.fit_transform(frame)
    piped = pipe.fit_transform(frame)
    assert_frame_equal(direct, piped)


def test_non_unique_index_and_multiindex_round_trip():
    frame = pd.DataFrame(
        {
            "e": [1, 1, 2, 2],
            "t": [1, 2, 1, 2],
            "x": [1.0, 2.0, 3.0, 4.0],
        },
        index=pd.MultiIndex.from_tuples([("a", 0), ("a", 1), ("b", 0), ("b", 1)]),
    )
    tr = GroupedLagFeatures(entity_variables="e", timestamp="t", periods=1)
    out = tr.fit_transform(frame)
    assert out.index.tolist() == frame.index.tolist()
    _assert_series_equal(out["x_lag_1"], [np.nan, 1.0, np.nan, 3.0])


def test_missing_values_ignore_propagates_nan():
    frame = pd.DataFrame(
        {
            "e": ["A", "A", "A"],
            "t": pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-03"]),
            "x": [1.0, np.nan, 3.0],
        }
    )
    tr = GroupedLagFeatures(
        entity_variables="e", timestamp="t", periods=1, missing_values="ignore"
    )
    out = tr.fit_transform(frame)
    assert out["x_lag_1"].iloc[1] == 1.0
    assert np.isnan(out["x_lag_1"].iloc[2])

    with pytest.raises(ValueError, match="NaN"):
        GroupedLagFeatures(entity_variables="e", timestamp="t", periods=1).fit(frame)


def test_date_format_parses_string_timestamps():
    frame = pd.DataFrame(
        {
            "e": ["A", "A"],
            "t": ["01/02/2020", "02/02/2020"],
            "x": [1.0, 2.0],
        }
    )
    tr = GroupedLagFeatures(
        entity_variables="e",
        timestamp="t",
        periods=1,
        date_format="%d/%m/%Y",
    )
    out = tr.fit_transform(frame)
    assert out["x_lag_1"].tolist()[1] == 1.0


def test_interleaved_entities_do_not_leak():
    frame = pd.DataFrame(
        {
            "store": ["A", "B", "A", "B", "A", "B"],
            "ts": pd.to_datetime(
                [
                    "2022-01-01",
                    "2022-01-01",
                    "2022-01-02",
                    "2022-01-02",
                    "2022-01-03",
                    "2022-01-03",
                ]
            ),
            "sales": [1.0, 100.0, 2.0, 200.0, 3.0, 300.0],
        }
    )
    tr = GroupedWindowFeatures(
        entity_variables="store", timestamp="ts", window=2, functions="sum"
    )
    out = tr.fit_transform(frame)

    # sorted per entity, each window only sees the same entity's values.
    # The default min_periods for an integer window is the window size, so the
    # second observation is still NaN.
    _assert_series_equal(
        out.loc[frame["store"] == "A", "sales_window_2_sum"],
        [np.nan, np.nan, 3.0],
    )
    _assert_series_equal(
        out.loc[frame["store"] == "B", "sales_window_2_sum"],
        [np.nan, np.nan, 300.0],
    )


def test_horizon_ignores_rows_inside_the_span():
    frame = pd.DataFrame(
        {
            "store": ["A", "A", "A", "A", "A"],
            "ts": pd.to_datetime(
                [
                    "2022-01-01",
                    "2022-01-02",
                    "2022-01-03",
                    "2022-01-04",
                    "2022-01-05",
                ]
            ),
            "sales": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    params = dict(
        entity_variables="store",
        timestamp="ts",
        window="2D",
        horizon="2D",
        functions="sum",
    )
    out = GroupedWindowFeatures(**params).fit_transform(frame)

    # changing 01-04 and 01-05 must not change the feature of 01-04, because
    # the 2D horizon caps history at 01-02.
    perturbed = frame.copy()
    perturbed.loc[perturbed["ts"] >= pd.Timestamp("2022-01-04"), "sales"] = 999.0
    out_perturbed = GroupedWindowFeatures(**params).fit_transform(perturbed)

    assert_frame_equal(
        out.loc[:2, ["sales_window_2D_sum"]],
        out_perturbed.loc[:2, ["sales_window_2D_sum"]],
    )
