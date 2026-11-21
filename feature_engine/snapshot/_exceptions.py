"""Exceptions raised by the feature snapshot, diff and rollback machinery."""


class SnapshotError(Exception):
    """Base class for all snapshot related errors."""


class SnapshotCorruptionError(SnapshotError):
    """Raised when a snapshot file cannot be parsed or fails schema validation.

    The message always states which segment of the file is broken.
    """

    def __init__(self, segment, reason):
        self.segment = segment
        self.reason = reason
        super().__init__(f"Corrupt snapshot in segment '{segment}': {reason}.")


class FeatureNameConflictError(SnapshotError):
    """Raised when two steps produce an output feature with the same name."""

    def __init__(self, feature_name, step_a, step_b=None):
        self.feature_name = feature_name
        self.step_a = step_a
        self.step_b = step_b
        if step_b is None:
            msg = (
                f"Feature name conflict: step '{step_a}' produces more than one "
                f"output column named '{feature_name}'."
            )
        else:
            msg = (
                f"Feature name conflict: steps '{step_a}' and '{step_b}' both "
                f"produce an output column named '{feature_name}'."
            )
        super().__init__(msg)


class SnapshotVersionError(SnapshotError):
    """Raised when a snapshot cannot be rebuilt with the installed library.

    The error carries migration advice explaining how to move the snapshot to a
    supported version.
    """

    def __init__(self, step_name, class_path, snapshot_version, reason, advice):
        self.step_name = step_name
        self.class_path = class_path
        self.snapshot_version = snapshot_version
        self.advice = advice
        super().__init__(
            f"Cannot restore step '{step_name}' ({class_path}) from snapshot "
            f"version '{snapshot_version}': {reason}. Migration advice: {advice}."
        )
