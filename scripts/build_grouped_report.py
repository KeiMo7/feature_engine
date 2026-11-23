"""Generate the grouped-forecasting report under output/.

Run from the repository root:

    python -m scripts.build_grouped_report

The script exercises three business lines with timestamp gaps, prints the
computed features next to a hand-computed reference, runs the grouped and the
pre-existing time-series test suites, and writes everything to ``output/``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engine.timeseries.forecasting import (
    GroupedExpandingWindowFeatures,
    GroupedLagFeatures,
    GroupedWindowFeatures,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"


def _demo_frame() -> pd.DataFrame:
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


def _manual_window(frame: pd.DataFrame, window: str, function: str) -> pd.Series:
    """Real-time interval [t - window, t) per entity, current row excluded."""
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    delta = pd.Timedelta(window)
    for _, block in frame.groupby("store", sort=False):
        block = block.sort_values("ts", kind="mergesort")
        times = pd.DatetimeIndex(block["ts"])
        values = block["sales"].to_numpy()
        for position, timestamp in zip(block.index, times):
            end = times.searchsorted(timestamp, side="left")
            start = times.searchsorted(timestamp - delta, side="left")
            segment = values[start:end]
            if len(segment):
                result.loc[position] = getattr(np, function)(segment)
    return result


def _section(title: str) -> str:
    return f"\n{title}\n{'-' * len(title)}\n"


def build_report() -> str:
    frame = _demo_frame()
    lines = ["Grouped forecasting features: output and verification report", "=" * 66]

    window = GroupedWindowFeatures(
        entity_variables="store",
        timestamp="ts",
        window="3D",
        functions=["mean", "sum"],
    )
    window_out = window.fit_transform(frame)
    expected_mean = _manual_window(frame, "3D", "mean")
    expected_sum = _manual_window(frame, "3D", "sum")
    matches = np.allclose(
        window_out["sales_window_3D_mean"].to_numpy(),
        expected_mean.to_numpy(),
        equal_nan=True,
    ) and np.allclose(
        window_out["sales_window_3D_sum"].to_numpy(),
        expected_sum.to_numpy(),
        equal_nan=True,
    )

    lines.append(_section("1. Three entities, 3D real-time window"))
    lines.append("Input:")
    lines.append(frame.to_string(index=False))
    lines.append("\nGroupedWindowFeatures(window='3D', functions=['mean', 'sum']):")
    lines.append(window_out.to_string(index=False))
    lines.append("\nHand-computed reference for sales (mean / sum):")
    reference = pd.DataFrame(
        {
            "store": frame["store"],
            "ts": frame["ts"].dt.strftime("%Y-%m-%d"),
            "expected_mean": expected_mean.to_numpy(),
            "expected_sum": expected_sum.to_numpy(),
        }
    )
    lines.append(reference.to_string(index=False))
    lines.append(f"\nMatches hand computation: {matches}")

    horizon = GroupedWindowFeatures(
        entity_variables="store",
        timestamp="ts",
        window="3D",
        functions="sum",
        horizon="1D",
    )
    horizon_out = horizon.fit_transform(frame)
    lines.append(_section("2. Forecast horizon ('1D')"))
    lines.append(
        horizon_out[["store", "ts", "sales", "sales_window_3D_sum"]].to_string(
            index=False
        )
    )
    lines.append(
        "\nFor a row at t, cut-off is t - 1D and the window is"
        " [cut-off - 3D, cut-off). The current row and future rows never"
        " contribute."
    )

    lags = GroupedLagFeatures(entity_variables="store", timestamp="ts", periods=[1, 2])
    lag_out = lags.fit_transform(frame)
    lines.append(_section("3. Lag features inside each entity"))
    lines.append(
        lag_out[["store", "ts", "sales", "sales_lag_1", "sales_lag_2"]].to_string(
            index=False
        )
    )

    expanding = GroupedExpandingWindowFeatures(
        entity_variables="store", timestamp="ts", functions=["mean", "count"]
    )
    expanding_out = expanding.fit_transform(frame)
    lines.append(_section("4. Expanding window per entity"))
    lines.append(
        expanding_out[
            ["store", "ts", "sales", "sales_expanding_mean", "sales_expanding_count"]
        ].to_string(index=False)
    )

    lines.append(_section("5. Cold start for an unseen entity"))
    new_frame = frame.copy()
    new_frame.loc[2, "store"] = "D"
    cold = GroupedWindowFeatures(
        entity_variables="store",
        timestamp="ts",
        window=2,
        cold_start="fill",
        cold_start_value=-1.0,
    )
    cold.fit(frame)
    cold_out = cold.transform(new_frame)
    default_cold = GroupedWindowFeatures(
        entity_variables="store", timestamp="ts", window=2
    )
    default_cold.fit(frame)
    try:
        default_cold.transform(new_frame)
        default_message = "no error (unexpected)"
    except ValueError as err:
        default_message = str(err).split(". Set")[0]
    lines.append(
        cold_out[["store", "ts", "sales", "sales_window_2_mean"]].to_string(index=False)
    )
    lines.append(f"\nDefault cold-start policy message: {default_message}")

    lines.append(_section("6. Duplicate timestamps"))
    duplicate_frame = pd.DataFrame(
        {
            "store": ["A", "A", "A", "A"],
            "ts": pd.to_datetime(
                ["2022-01-01", "2022-01-02", "2022-01-02", "2022-01-03"]
            ),
            "sales": [1.0, 2.0, 4.0, 8.0],
        }
    )
    keep = GroupedLagFeatures(
        entity_variables="store", timestamp="ts", periods=1, duplicates="keep"
    )
    lines.append(keep.fit_transform(duplicate_frame).to_string(index=False))
    raise_error = GroupedLagFeatures(entity_variables="store", timestamp="ts")
    try:
        raise_error.fit(duplicate_frame)
        duplicate_message = "no error (unexpected)"
    except ValueError as err:
        duplicate_message = str(err)
    lines.append(f"\nDefault duplicates='raise' message: {duplicate_message}")

    return "\n".join(lines) + "\n"


def run_tests() -> str:
    report = []
    commands = [
        (
            "Grouped forecasting tests",
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_time_series/test_forecasting/"
                "test_grouped_forecast_features.py",
                "-q",
            ],
        ),
        (
            "Pre-existing time-series suite",
            [sys.executable, "-m", "pytest", "tests/test_time_series", "-q"],
        ),
    ]
    for title, command in commands:
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        report.append(_section(title))
        report.append(completed.stdout)
        if completed.stderr:
            report.append(completed.stderr)
        report.append(f"Exit code: {completed.returncode}\n")
    return "\n".join(report)


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    report = build_report()
    (OUTPUT / "grouped_forecast_report.md").write_text(report)
    tests = run_tests()
    (OUTPUT / "grouped_forecast_tests.txt").write_text(tests)
    print(report)
    print(tests)


if __name__ == "__main__":
    main()
