"""Opt-in validation of the statistics learned by transformers.

This subpackage pins down and verifies how the statistics used by
Feature-engine transformers (means, medians, quantiles, ...) are computed:

- :class:`StatisticsSnapshot` records, at fit time, the value of every
  learned statistic, the data range it was computed from (full, sampled or
  chunked) and whether it is exact or approximate (with error bounds).
- :class:`StatisticsValidationMixin` freezes the statistics after fit
  (accidental re-fits raise instead of silently changing them) and provides
  train-vs-inference consistency checks and distribution drift detection.
- :class:`ChunkedStatisticsAccumulator` computes exact (mean, variance) and
  approximate (quantiles, with DKW error bounds) statistics over data chunks.

The validation is opt-in: transformers behave exactly as before unless their
``validate_stats`` parameter is set to True.
"""

from feature_engine.validation._accumulator import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESERVOIR_SIZE,
    ChunkedStatisticsAccumulator,
)
from feature_engine.validation._exceptions import (
    DistributionDriftWarning,
    InsufficientSampleWarning,
    MissingStatisticsError,
    StatisticsMismatchWarning,
    StatisticsRefitError,
    StatisticsValidationWarning,
)
from feature_engine.validation._mixin import (
    DEFAULT_DRIFT_THRESHOLD,
    DEFAULT_MIN_DRIFT_SAMPLES,
    StatisticsValidationMixin,
)
from feature_engine.validation._snapshot import StatisticsSnapshot

__all__ = [
    "ChunkedStatisticsAccumulator",
    "DistributionDriftWarning",
    "InsufficientSampleWarning",
    "MissingStatisticsError",
    "StatisticsMismatchWarning",
    "StatisticsRefitError",
    "StatisticsSnapshot",
    "StatisticsValidationMixin",
    "StatisticsValidationWarning",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_DRIFT_THRESHOLD",
    "DEFAULT_MIN_DRIFT_SAMPLES",
    "DEFAULT_RESERVOIR_SIZE",
]
