"""
The module snapshot contains the feature snapshot, diff and rollback machinery.

Snapshots describe feature-set *structure* — names, types, lineage and the
parameters of every transformation step — but never contain rows of data.
"""

from feature_engine.snapshot._diff import SnapshotDiff, diff_snapshots
from feature_engine.snapshot._exceptions import (
    FeatureNameConflictError,
    SnapshotCorruptionError,
    SnapshotError,
    SnapshotVersionError,
)
from feature_engine.snapshot._rollback import restore_pipeline, rollback
from feature_engine.snapshot._snapshot import FeatureSnapshot
from feature_engine.snapshot._tracking import build_snapshot

__all__ = [
    "FeatureSnapshot",
    "SnapshotDiff",
    "FeatureNameConflictError",
    "SnapshotCorruptionError",
    "SnapshotError",
    "SnapshotVersionError",
    "build_snapshot",
    "diff_snapshots",
    "restore_pipeline",
    "rollback",
]
