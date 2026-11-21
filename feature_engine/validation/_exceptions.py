"""Exceptions and warnings raised by the statistics validation utilities.

These classes are part of the opt-in statistics validation capability. None of
them are raised or emitted when the validation is switched off (the default).
"""


class StatisticsRefitError(RuntimeError):
    """Raised when a transformer with frozen statistics is fitted again.

    Statistics learned during ``fit()`` are frozen once fitting completes when
    the validation is enabled. Calling ``fit()`` again would silently replace
    them, so it raises this error instead. Use ``refit()`` to explicitly
    re-learn the statistics from new data.
    """


class MissingStatisticsError(RuntimeError):
    """Raised when the statistics needed at inference time are not available.

    Raised at ``transform()`` when the transformer was fitted without the
    statistics validation enabled (or the recorded statistics were lost, for
    example when unpickling a model saved by an older version), so there is no
    frozen statistics snapshot to validate against. The transformer never
    falls back to default values.
    """


class StatisticsMismatchWarning(UserWarning):
    """Emitted when the train and inference statistics differ beyond tolerance."""


class DistributionDriftWarning(UserWarning):
    """Emitted when the inference data distribution drifts from the train one."""


class InsufficientSampleWarning(UserWarning):
    """Emitted when there is not enough data to draw a drift conclusion."""


class StatisticsValidationWarning(UserWarning):
    """Emitted when a validation check itself fails.

    The validation utilities never block inference: if a check cannot be
    completed, this warning is emitted instead of raising an exception.
    """
