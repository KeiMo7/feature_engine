"""Opt-in statistics validation for transformers that learn statistics.

The :class:`StatisticsValidationMixin` adds, to any transformer that learns
statistics from data (means, medians, quantiles, ...):

- **provenance**: every fit records which data range the statistics were
  learned from (``"full"``, ``"sampled"`` or ``"chunked"``) and whether they
  are exact or approximate (with error bounds),
- **freezing**: once fitted, calling ``fit()`` again raises
  :class:`StatisticsRefitError` instead of silently replacing the statistics;
  use ``refit()`` to explicitly re-learn them,
- **consistency checks**: ``check_statistics()`` compares the frozen
  statistics against another fitted transformer, a saved snapshot, or fresh
  data, and warns when they differ beyond tolerance,
- **drift detection**: ``check_drift()`` compares the distribution of new
  data against the distribution seen at fit time and warns on drift.

Everything is opt-in through the ``validate_stats`` init parameter of the
hosting transformer. When ``validate_stats=False`` (the default), none of the
checks run and the transformer behaves exactly as before.
"""

import warnings
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from feature_engine.validation._exceptions import (
    DistributionDriftWarning,
    InsufficientSampleWarning,
    MissingStatisticsError,
    StatisticsMismatchWarning,
    StatisticsRefitError,
    StatisticsValidationWarning,
)
from feature_engine.validation._snapshot import (
    HISTOGRAM_BINS,
    QUANTILE_GRID,
    StatisticsSnapshot,
)

# Minimum fraction assigned to empty histogram bins when computing PSI, aligned
# with DropHighPSIFeatures (selection module).
_MIN_BIN_PCT = 0.0001

DEFAULT_DRIFT_THRESHOLD = 0.25
DEFAULT_MIN_DRIFT_SAMPLES = 200


class StatisticsValidationMixin:
    """Mixin with the statistics validation functionality.

    The hosting transformer must:

    - accept and store a ``validate_stats`` boolean init parameter,
    - call ``self._guard_against_refit()`` at the top of ``fit()``,
    - call ``self._record_fit_statistics(...)`` at the end of ``fit()``,
    - implement ``_get_fitted_statistics()`` returning
      ``{variable: {statistic_name: value}}``.
    """

    # -- internal helpers -------------------------------------------------

    def _stats_validation_on(self) -> bool:
        """True when the statistics validation is enabled."""
        return bool(getattr(self, "validate_stats", False))

    def _is_fitted(self) -> bool:
        try:
            check_is_fitted(self)
            return True
        except NotFittedError:
            return False

    def _guard_against_refit(self) -> None:
        """Raise if the transformer was already fitted.

        No-op when the validation is off or when the fit was explicitly
        requested through ``refit()``.
        """
        if not self._stats_validation_on():
            return
        if getattr(self, "_allow_refit", False):
            return
        if self._is_fitted():
            raise StatisticsRefitError(
                f"This {type(self).__name__} was already fitted and its "
                "statistics are frozen (validate_stats=True). Calling fit() "
                "again would silently replace them. If you really want to "
                "re-learn the statistics from new data, call refit(X) "
                "instead. To disable the guard, set validate_stats=False."
            )

    def _reset_stats_validation_state(self) -> None:
        """Drop the recorded statistics and any chunked-fitting state."""
        for attr in ("statistics_snapshot_", "_stats_accumulator_"):
            if hasattr(self, attr):
                delattr(self, attr)

    def _check_statistics_available(self) -> None:
        """Raise if the frozen statistics are missing at inference time.

        No-op when the validation is off. Never falls back to default
        values: a missing snapshot is an explicit error.
        """
        if not self._stats_validation_on():
            return
        if getattr(self, "statistics_snapshot_", None) is None:
            raise MissingStatisticsError(
                f"This {type(self).__name__} has no recorded statistics, but "
                "validate_stats=True. Fit it with validate_stats=True (or "
                "partial_fit) so the statistics snapshot is recorded, or set "
                "validate_stats=False to disable the validation."
            )

    def _get_drift_variables(self):
        """Variables monitored for drift. Defaults to ``variables_``."""
        return self.variables_

    def _get_fitted_statistics(self) -> Dict:
        """Statistics learned during fit, as {variable: {name: value}}."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _get_fitted_statistics() "
            "to use the statistics validation."
        )

    # -- recording ---------------------------------------------------------

    def _compute_distribution_summary(
        self, X: pd.DataFrame, variables, bins: int = HISTOGRAM_BINS
    ) -> Dict:
        """Compact per-variable distribution summary (quantiles + histogram).

        Only the summary is stored; the raw data is never kept.
        """
        summary = {}
        for v in variables:
            values = X[v].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            quantiles = {str(q): float(np.quantile(values, q)) for q in QUANTILE_GRID}
            lo, hi = float(values.min()), float(values.max())
            if lo == hi:
                # constant variable: widen the range so the histogram is valid
                lo, hi = lo - 0.5, hi + 0.5
            edges = np.linspace(lo, hi, bins + 1)
            counts, _ = np.histogram(values, bins=edges)
            summary[v] = {
                "quantiles": quantiles,
                "hist_counts": [int(c) for c in counts],
                "hist_edges": [float(e) for e in edges],
            }
        return summary

    def _record_fit_statistics(
        self,
        X: Optional[pd.DataFrame],
        *,
        statistics: Dict,
        scope: str,
        exact: Optional[Dict] = None,
        error_bounds: Optional[Dict] = None,
        error_bound_confidence: Optional[float] = None,
        n_chunks: int = 1,
        random_state: Optional[int] = None,
        distribution_summary: Optional[Dict] = None,
        n_samples: Optional[int] = None,
        force: bool = False,
    ) -> None:
        """Record the statistics learned during fit in a snapshot.

        No-op unless the validation is enabled or ``force=True`` (used by
        ``partial_fit()``, which always records its chunked provenance).
        """
        if not (force or self._stats_validation_on()):
            return
        if distribution_summary is None and X is not None:
            distribution_summary = self._compute_distribution_summary(
                X, self._get_drift_variables()
            )
        if n_samples is None:
            n_samples = len(X) if X is not None else 0
        self.statistics_snapshot_ = StatisticsSnapshot(
            statistics=statistics,
            n_samples=n_samples,
            scope=scope,
            transformer=type(self).__name__,
            exact=exact,
            error_bounds=error_bounds,
            error_bound_confidence=error_bound_confidence,
            n_chunks=n_chunks,
            random_state=random_state,
            distribution_summary=distribution_summary,
        )

    # -- public API ---------------------------------------------------------

    def refit(self, X: pd.DataFrame, y=None):
        """Explicitly re-learn the statistics from new data.

        Identical to ``fit()``, but allowed when the statistics are frozen.
        Use it when the statistics *should* change, so that accidental
        ``fit()`` calls at inference time keep raising an error.
        """
        self._allow_refit = True
        try:
            self._reset_stats_validation_state()
            return self.fit(X, y)
        finally:
            self._allow_refit = False

    def save_statistics(self, path: Union[str, Path]) -> None:
        """Save the recorded statistics snapshot to a JSON file.

        The saved snapshot is enough to run ``check_statistics()`` later, for
        example in the inference environment, without loading the training
        data.
        """
        snapshot = getattr(self, "statistics_snapshot_", None)
        if snapshot is None:
            raise MissingStatisticsError(
                f"This {type(self).__name__} has no recorded statistics. "
                "Fit it with validate_stats=True (or partial_fit) first."
            )
        snapshot.to_json(path)

    def _coerce_other_snapshot(self, other) -> StatisticsSnapshot:
        """Build a snapshot from a transformer, snapshot, dict or JSON path."""
        if isinstance(other, StatisticsSnapshot):
            return other
        if isinstance(other, dict):
            return StatisticsSnapshot.from_dict(other)
        if isinstance(other, (str, Path)):
            return StatisticsSnapshot.from_json(other)
        snapshot = getattr(other, "statistics_snapshot_", None)
        if snapshot is not None:
            return snapshot
        if hasattr(other, "_get_fitted_statistics"):
            return StatisticsSnapshot(
                statistics=other._get_fitted_statistics(),
                n_samples=0,
                scope="full",
                transformer=type(other).__name__,
            )
        raise TypeError(
            "other must be a StatisticsSnapshot, a dict, a path to a JSON "
            f"snapshot, or a fitted transformer. Got {type(other)} instead."
        )

    def check_statistics(
        self,
        other=None,
        X: Optional[pd.DataFrame] = None,
        rtol: float = 1e-3,
        atol: float = 1e-9,
    ) -> Dict:
        """Compare the frozen statistics against another source.

        Parameters
        ----------
        other: transformer, StatisticsSnapshot, dict or path, default=None
            What to compare against: another fitted transformer, a snapshot
            (or its dictionary / JSON file) saved with ``save_statistics()``.
            Exactly one of ``other`` and ``X`` must be given.

        X: pandas dataframe, default=None
            Fresh data (e.g. the inference data). The statistics are
            re-learned on it with the current parameters and compared against
            the frozen ones.

        rtol: float, default=1e-3
            Relative tolerance. A statistic differs when
            ``abs(a - b) > atol + rtol * abs(a)``.

        atol: float, default=1e-9
            Absolute tolerance.

        Returns
        -------
        report: dict
            JSON-serializable comparison report with both values of every
            statistic. ``report["status"]`` is ``"ok"``, ``"mismatch"`` or
            ``"error"``. A :class:`StatisticsMismatchWarning` is emitted when
            the statistics differ beyond tolerance. If the check itself
            fails, a :class:`StatisticsValidationWarning` is emitted and the
            report has status ``"error"``; the check never raises (except
            :class:`MissingStatisticsError` when there are no frozen
            statistics to compare).
        """
        snapshot = getattr(self, "statistics_snapshot_", None)
        if snapshot is None:
            raise MissingStatisticsError(
                f"This {type(self).__name__} has no recorded statistics. "
                "Fit it with validate_stats=True (or partial_fit) first."
            )
        if (other is None) == (X is None):
            raise ValueError("Pass exactly one of `other` or `X`.")
        try:
            if X is not None:
                fresh = clone(self)
                if "validate_stats" in fresh.get_params():
                    fresh.set_params(validate_stats=False)
                fresh.fit(X)
                other_snapshot = StatisticsSnapshot(
                    statistics=fresh._get_fitted_statistics(),
                    n_samples=len(X),
                    scope="full",
                    transformer=type(fresh).__name__,
                )
            else:
                other_snapshot = self._coerce_other_snapshot(other)
            report = snapshot.compare(other_snapshot, rtol=rtol, atol=atol)
        except Exception as exc:  # the check must never break inference
            warnings.warn(
                f"The statistics check failed: {exc}",
                StatisticsValidationWarning,
            )
            return {"status": "error", "error": str(exc)}

        if report["status"] == "mismatch":
            details = []
            for variable, var_report in report["variables"].items():
                for name, entry in var_report["statistics"].items():
                    if not entry["within_tolerance"]:
                        details.append(
                            f"{variable}.{name}: fit={entry['fit']!r} vs "
                            f"other={entry['other']!r}"
                        )
            warnings.warn(
                f"The statistics of {type(self).__name__} differ beyond "
                f"tolerance (rtol={rtol}, atol={atol}): " + "; ".join(details),
                StatisticsMismatchWarning,
            )
        return report

    def check_drift(
        self,
        X: pd.DataFrame,
        transformed: bool = False,
        threshold: float = DEFAULT_DRIFT_THRESHOLD,
        min_samples: int = DEFAULT_MIN_DRIFT_SAMPLES,
    ) -> Dict:
        """Compare the distribution of new data against the fit-time one.

        Uses the Population Stability Index (PSI) per variable, computed on
        the histogram stored in the statistics snapshot (the same binning and
        PSI formula as :class:`feature_engine.selection.DropHighPSIFeatures`),
        plus the maximum quantile shift relative to the fit-time IQR.

        Parameters
        ----------
        X: pandas dataframe
            The new (e.g. inference) data.

        transformed: bool, default=False
            When True, the data is transformed with ``transform()`` before
            the comparison, so the check applies to the features the model
            actually sees.

        threshold: float, default=0.25
            PSI above this value flags drift (0.25 is the common rule of
            thumb for significant population shift).

        min_samples: int, default=200
            Minimum number of rows required to draw a conclusion. With fewer
            rows, an :class:`InsufficientSampleWarning` is emitted and the
            report status is ``"insufficient_samples"`` (no verdict).

        Returns
        -------
        report: dict
            JSON-serializable report; ``report["status"]`` is ``"ok"``,
            ``"drift"``, ``"insufficient_samples"`` or ``"error"``. Drift
            emits a :class:`DistributionDriftWarning`. If the check itself
            fails, a :class:`StatisticsValidationWarning` is emitted instead
            of raising.
        """
        snapshot = getattr(self, "statistics_snapshot_", None)
        if snapshot is None:
            raise MissingStatisticsError(
                f"This {type(self).__name__} has no recorded statistics. "
                "Fit it with validate_stats=True (or partial_fit) first."
            )
        try:
            if transformed:
                X = self.transform(X)
            n_samples = len(X)
            if n_samples < min_samples:
                warnings.warn(
                    f"Drift check needs at least {min_samples} rows, got "
                    f"{n_samples}. The sample is too small to conclude "
                    "whether the distribution drifted.",
                    InsufficientSampleWarning,
                )
                return {
                    "status": "insufficient_samples",
                    "n_samples": n_samples,
                    "min_samples": min_samples,
                }

            variables_report = {}
            drifted = []
            for variable, summary in snapshot.distribution_summary.items():
                values = X[variable].dropna().to_numpy(dtype=float)
                if values.size == 0:
                    continue
                edges = np.asarray(summary["hist_edges"], dtype=float)
                # extend the outer bins so out-of-range values are counted
                open_edges = edges.copy()
                open_edges[0], open_edges[-1] = -np.inf, np.inf
                new_counts, _ = np.histogram(values, bins=open_edges)
                psi = _psi(
                    np.asarray(summary["hist_counts"], dtype=float),
                    new_counts.astype(float),
                )
                fit_q = summary["quantiles"]
                iqr = fit_q["0.75"] - fit_q["0.25"]
                scale = iqr if iqr > 0 else 1.0
                shift = max(
                    abs(float(np.quantile(values, float(q))) - v) / scale
                    for q, v in fit_q.items()
                )
                variables_report[variable] = {
                    "psi": psi,
                    "max_quantile_shift": shift,
                    "drift": psi > threshold,
                }
                if psi > threshold:
                    drifted.append(f"{variable} (PSI={psi:.4f})")

            status = "drift" if drifted else "ok"
            report = {
                "status": status,
                "threshold": threshold,
                "min_samples": min_samples,
                "n_samples": n_samples,
                "transformed": transformed,
                "variables": variables_report,
            }
        except Exception as exc:  # the check must never break inference
            warnings.warn(
                f"The drift check failed: {exc}",
                StatisticsValidationWarning,
            )
            return {"status": "error", "error": str(exc)}

        if status == "drift":
            warnings.warn(
                f"The distribution of {type(self).__name__} input data "
                f"drifted from the one seen at fit time (PSI > {threshold}): "
                + ", ".join(drifted),
                DistributionDriftWarning,
            )
        return report


def _psi(fit_counts: np.ndarray, new_counts: np.ndarray) -> float:
    """Population Stability Index between two binned distributions.

    Same formula as DropHighPSIFeatures: empty bins are replaced by a small
    fraction so the logarithm is always defined.
    """
    fit_pct = fit_counts / fit_counts.sum()
    new_pct = new_counts / new_counts.sum()
    fit_pct = np.clip(fit_pct, _MIN_BIN_PCT, None)
    new_pct = np.clip(new_pct, _MIN_BIN_PCT, None)
    return float(np.sum((new_pct - fit_pct) * np.log(new_pct / fit_pct)))
