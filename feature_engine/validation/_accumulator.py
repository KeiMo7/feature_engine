"""Exact and approximate streaming statistics over data chunks.

The :class:`ChunkedStatisticsAccumulator` aggregates statistics over a dataset
that arrives in several chunks (``partial_fit``), without holding the full
dataset in memory:

- count, mean and variance are **exact** (up to floating-point summation
  order): they are computed from running sums.
- quantiles are **approximate**: they are estimated from a reservoir sample
  of fixed size per variable. The error bound is derived from the
  Dvoretzky-Kiefer-Wolfowitz (DKW) inequality and recorded together with its
  confidence level.

Determinism: for a fixed ``random_state``, the same sequence of chunks (same
rows, in the same order, split at the same boundaries) always produces the
same reservoir and therefore the same approximate quantiles. No
multiprocessing is involved, so the number of processes has no effect on the
result. With ``random_state=None`` the approximate quantiles are not
reproducible run-to-run; the exact statistics (count, mean, variance) never
depend on the seed.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_RESERVOIR_SIZE = 1024
DEFAULT_CONFIDENCE = 0.95


class ChunkedStatisticsAccumulator:
    """Accumulate exact and approximate statistics over data chunks.

    Parameters
    ----------
    variables: list
        The variables to accumulate statistics for.

    reservoir_size: int, default=1024
        Maximum number of values kept per variable to estimate quantiles.
        Bigger reservoirs give tighter error bounds at a higher memory cost
        (``reservoir_size`` floats per variable, independent of the data
        size).

    random_state: int or None, default=None
        Seed for the reservoir sampling. Fix it to make the approximate
        quantiles reproducible.
    """

    def __init__(
        self,
        variables: List,
        reservoir_size: int = DEFAULT_RESERVOIR_SIZE,
        random_state: Optional[int] = None,
    ):
        if not isinstance(reservoir_size, int) or reservoir_size < 10:
            raise ValueError(
                "reservoir_size must be an integer >= 10. "
                f"Got {reservoir_size} instead."
            )
        self.variables = list(variables)
        self.reservoir_size = reservoir_size
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self.n_chunks = 0
        self.n_rows = 0
        self._count = {v: 0 for v in self.variables}
        self._sum = {v: 0.0 for v in self.variables}
        self._sumsq = {v: 0.0 for v in self.variables}
        self._min = {v: np.inf for v in self.variables}
        self._max = {v: -np.inf for v in self.variables}
        self._reservoir = {v: [] for v in self.variables}

    @property
    def n_samples(self) -> int:
        """Number of non-missing observations seen per variable."""
        return self._count

    def update(self, X: pd.DataFrame) -> "ChunkedStatisticsAccumulator":
        """Update the accumulated statistics with a new chunk of data."""
        for v in self.variables:
            values = X[v].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            self._count[v] += values.size
            self._sum[v] += values.sum()
            self._sumsq[v] += np.square(values).sum()
            self._min[v] = min(self._min[v], values.min())
            self._max[v] = max(self._max[v], values.max())
            self._update_reservoir(v, values)
        self.n_chunks += 1
        self.n_rows += len(X)
        return self

    def _update_reservoir(self, variable, values: np.ndarray) -> None:
        """Vectorized reservoir sampling (algorithm R).

        Each incoming value is accepted with probability k / t, where k is the
        reservoir size and t the number of values seen so far; accepted values
        replace a uniformly chosen slot. Statistically equivalent to the
        sequential algorithm, deterministic for a fixed seed.
        """
        reservoir = self._reservoir[variable]
        k = self.reservoir_size
        seen = self._count[variable]
        n = values.size
        first = seen - n  # values seen before this batch

        missing = k - len(reservoir)
        if missing > 0:
            take = min(missing, n)
            reservoir.extend(values[:take].tolist())
            values = values[take:]
            first += take
            n = values.size
            if n == 0:
                return

        t = first + np.arange(1, n + 1)
        accept = self._rng.random(n) < (k / t)
        n_accepted = int(accept.sum())
        if n_accepted:
            slots = self._rng.integers(0, k, size=n_accepted)
            accepted_values = values[accept]
            res_array = np.asarray(reservoir)
            res_array[slots] = accepted_values
            self._reservoir[variable] = res_array.tolist()

    def mean(self, variable) -> float:
        """Exact mean of a variable (from running sums)."""
        if self._count[variable] == 0:
            return np.nan
        return self._sum[variable] / self._count[variable]

    def std(self, variable) -> float:
        """Exact population standard deviation (ddof=0) of a variable."""
        n = self._count[variable]
        if n == 0:
            return np.nan
        mean = self.mean(variable)
        variance = self._sumsq[variable] / n - mean**2
        return math.sqrt(max(variance, 0.0))

    def min(self, variable) -> float:
        """Exact minimum observed for a variable."""
        return self._min[variable]

    def max(self, variable) -> float:
        """Exact maximum observed for a variable."""
        return self._max[variable]

    def quantile(self, variable, q: float) -> float:
        """Approximate quantile of a variable, from the reservoir sample."""
        reservoir = self._reservoir[variable]
        if not reservoir:
            return np.nan
        return float(np.quantile(np.asarray(reservoir), q))

    def quantile_error_bound(
        self, variable, q: float, confidence: float = DEFAULT_CONFIDENCE
    ) -> float:
        """Error bound of the approximate quantile, in value units.

        Uses the Dvoretzky-Kiefer-Wolfowitz inequality: for a reservoir of k
        values, the empirical CDF differs from the true CDF by at most
        ``eps = sqrt(ln(2 / (1 - confidence)) / (2k))`` everywhere, with
        probability at least ``confidence``. The true q-quantile therefore
        lies between the (q - eps) and (q + eps) quantiles of the reservoir,
        which gives a bound in value units.

        Returns np.nan when the reservoir is empty.
        """
        reservoir = self._reservoir[variable]
        if not reservoir:
            return np.nan
        k = len(reservoir)
        eps = math.sqrt(math.log(2.0 / (1.0 - confidence)) / (2.0 * k))
        sample = np.asarray(reservoir)
        q_hat = float(np.quantile(sample, q))
        q_lo = float(np.quantile(sample, max(0.0, q - eps)))
        q_hi = float(np.quantile(sample, min(1.0, q + eps)))
        return max(q_hi - q_hat, q_hat - q_lo)

    def quantiles_with_bounds(
        self,
        quantiles: Sequence[float],
        confidence: float = DEFAULT_CONFIDENCE,
    ) -> Tuple[Dict, Dict]:
        """Approximate quantiles and their error bounds for all variables.

        Returns
        -------
        values: dict
            ``{variable: {quantile_label: value}}`` with the estimates.

        bounds: dict
            ``{variable: {quantile_label: bound}}`` with the error bounds,
            valid with probability at least ``confidence``.
        """
        values, bounds = {}, {}
        for v in self.variables:
            values[v] = {}
            bounds[v] = {}
            for q in quantiles:
                label = f"q{q}"
                values[v][label] = self.quantile(v, q)
                bounds[v][label] = self.quantile_error_bound(v, q, confidence)
        return values, bounds

    def distribution_summary(self, variable, bins: int = 10) -> Dict:
        """Compact distribution summary of a variable, from the reservoir.

        Approximate: built from the reservoir sample instead of the full
        data. Used for drift detection after chunked fitting.
        """
        reservoir = np.asarray(self._reservoir[variable])
        if reservoir.size == 0:
            return {"quantiles": {}, "hist_counts": [], "hist_edges": []}
        quantiles = {
            str(q): float(np.quantile(reservoir, q))
            for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
        }
        edges = np.linspace(reservoir.min(), reservoir.max(), bins + 1)
        counts, _ = np.histogram(reservoir, bins=edges)
        return {
            "quantiles": quantiles,
            "hist_counts": counts.tolist(),
            "hist_edges": edges.tolist(),
        }
