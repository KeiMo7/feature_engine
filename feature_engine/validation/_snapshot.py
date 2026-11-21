"""Immutable record of the statistics learned by a transformer during fit.

The :class:`StatisticsSnapshot` captures, for a fitted transformer:

- the value of every learned statistic (e.g. the mean or median per variable),
- the data scope used to compute them (``"full"``, ``"sampled"`` or
  ``"chunked"``),
- whether each statistic is exact or approximate, and the error bound of the
  approximate ones,
- a compact distribution summary (quantile grid and histogram) used later for
  drift detection.

Snapshots are small (they never store raw data) and serializable to JSON, so
the statistics consistency between train and inference can be checked without
reloading the full training data.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Union

# Quantile grid stored in every snapshot for drift detection.
QUANTILE_GRID = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)

# Number of equal-width bins of the histogram stored in every snapshot.
HISTOGRAM_BINS = 10

VALID_SCOPES = ("full", "sampled", "chunked")


class StatisticsSnapshot:
    """Record of the statistics learned during fit.

    Parameters
    ----------
    statistics: dict
        Nested dictionary ``{variable: {statistic_name: value}}`` with the
        learned statistics, for example ``{"age": {"mean": 35.2}}``.

    n_samples: int
        Number of observations used to learn the statistics.

    scope: str
        Data range used at fit time. One of ``"full"`` (the complete training
        set), ``"sampled"`` (a user-drawn sample) or ``"chunked"`` (the data
        was fed in several chunks through ``partial_fit()``).

    transformer: str, default=""
        Name of the transformer class that produced the snapshot.

    exact: dict, default=None
        ``{statistic_name: bool}`` indicating whether each statistic is exact
        or approximate. Defaults to all exact.

    error_bounds: dict, default=None
        ``{variable: {statistic_name: bound}}`` with the error bound of each
        approximate statistic. Empty for exact statistics.

    error_bound_confidence: float, default=None
        Confidence level of the error bounds (e.g. 0.95). Only meaningful for
        approximate statistics.

    n_chunks: int, default=1
        Number of data chunks the statistics were learned from. 1 unless the
        scope is ``"chunked"``.

    random_state: int or None, default=None
        Seed of the random number generator used by approximate (sampling
        based) statistics. ``None`` means the statistics are not
        reproducible run-to-run.

    distribution_summary: dict, default=None
        ``{variable: {"quantiles": {...}, "hist_counts": [...],
        "hist_edges": [...]}}`` used for drift detection.
    """

    def __init__(
        self,
        statistics: Dict,
        n_samples: int,
        scope: str,
        transformer: str = "",
        exact: Optional[Dict] = None,
        error_bounds: Optional[Dict] = None,
        error_bound_confidence: Optional[float] = None,
        n_chunks: int = 1,
        random_state: Optional[int] = None,
        distribution_summary: Optional[Dict] = None,
    ):
        if scope not in VALID_SCOPES:
            raise ValueError(
                f"scope must be one of {VALID_SCOPES}. Got {scope} instead."
            )
        self.statistics = statistics
        self.n_samples = int(n_samples)
        self.scope = scope
        self.transformer = transformer
        self.exact = exact if exact is not None else {}
        self.error_bounds = error_bounds if error_bounds is not None else {}
        self.error_bound_confidence = error_bound_confidence
        self.n_chunks = int(n_chunks)
        self.random_state = random_state
        self.distribution_summary = (
            distribution_summary if distribution_summary is not None else {}
        )
        self.created_at = datetime.now(timezone.utc).isoformat()

    def is_exact(self, statistic: str) -> bool:
        """Return True if the given statistic was computed exactly."""
        return self.exact.get(statistic, True)

    def to_dict(self) -> Dict:
        """Return a JSON-serializable dictionary with the snapshot content."""
        return {
            "statistics": self.statistics,
            "n_samples": self.n_samples,
            "scope": self.scope,
            "transformer": self.transformer,
            "exact": self.exact,
            "error_bounds": self.error_bounds,
            "error_bound_confidence": self.error_bound_confidence,
            "n_chunks": self.n_chunks,
            "random_state": self.random_state,
            "distribution_summary": self.distribution_summary,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "StatisticsSnapshot":
        """Rebuild a snapshot from a dictionary produced by ``to_dict()``."""
        snapshot = cls(
            statistics=data["statistics"],
            n_samples=data["n_samples"],
            scope=data["scope"],
            transformer=data.get("transformer", ""),
            exact=data.get("exact"),
            error_bounds=data.get("error_bounds"),
            error_bound_confidence=data.get("error_bound_confidence"),
            n_chunks=data.get("n_chunks", 1),
            random_state=data.get("random_state"),
            distribution_summary=data.get("distribution_summary"),
        )
        if "created_at" in data:
            snapshot.created_at = data["created_at"]
        return snapshot

    def to_json(self, path: Union[str, Path]) -> None:
        """Save the snapshot to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "StatisticsSnapshot":
        """Load a snapshot from a JSON file created with ``to_json()``."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def compare(
        self, other: "StatisticsSnapshot", rtol: float = 1e-3, atol: float = 1e-9
    ) -> Dict:
        """Compare the statistics of this snapshot against another one.

        Parameters
        ----------
        other: StatisticsSnapshot
            The snapshot to compare against (e.g. the one recorded at
            inference time).

        rtol: float, default=1e-3
            Relative tolerance. A statistic differs when
            ``abs(a - b) > atol + rtol * abs(b)``.

        atol: float, default=1e-9
            Absolute tolerance.

        Returns
        -------
        report: dict
            A JSON-serializable dictionary with the comparison of every
            statistic of every variable: both values, the absolute difference
            and whether it is within tolerance. The report ``status`` is
            ``"ok"`` when every statistic matches and ``"mismatch"``
            otherwise.
        """
        variables_report = {}
        overall_status = "ok"
        for variable, stats in self.statistics.items():
            other_stats = other.statistics.get(variable, {})
            stat_report = {}
            var_status = "ok"
            for name, fit_value in stats.items():
                if name not in other_stats:
                    entry = {
                        "fit": fit_value,
                        "other": None,
                        "abs_diff": None,
                        "within_tolerance": False,
                        "note": "statistic missing in the other snapshot",
                    }
                    var_status = "mismatch"
                else:
                    other_value = other_stats[name]
                    abs_diff = abs(fit_value - other_value)
                    within = abs_diff <= atol + rtol * abs(fit_value)
                    entry = {
                        "fit": fit_value,
                        "other": other_value,
                        "abs_diff": abs_diff,
                        "within_tolerance": within,
                    }
                    if not within:
                        var_status = "mismatch"
                stat_report[name] = entry
            if var_status == "mismatch":
                overall_status = "mismatch"
            variables_report[variable] = {
                "status": var_status,
                "statistics": stat_report,
            }

        missing_variables = set(other.statistics) - set(self.statistics)
        if missing_variables:
            overall_status = "mismatch"

        return {
            "status": overall_status,
            "transformer": self.transformer,
            "scope": self.scope,
            "n_samples_fit": self.n_samples,
            "n_samples_other": other.n_samples,
            "tolerances": {"rtol": rtol, "atol": atol},
            "variables": variables_report,
            "variables_only_in_other": sorted(missing_variables),
            "compared_at": datetime.now(timezone.utc).isoformat(),
        }
