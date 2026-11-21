"""Differences between two feature snapshots.

Four feature-level categories are reported separately — added, removed,
renamed and dtype changed — plus step parameter changes so that a chain
edit is fully auditable.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Union

from feature_engine.snapshot._snapshot import FeatureSnapshot


@dataclass
class SnapshotDiff:
    """Result of comparing two :class:`FeatureSnapshot` objects.

    Attributes
    ----------
    added: list of str
        Names of features present in the new snapshot but with no matching
        identity in the old one.

    removed: list of str
        Names of features present in the old snapshot but with no matching
        identity in the new one.

    renamed: list of dict
        One entry per renamed feature: ``{"old_name", "new_name", "identity"}``.
        Rename detection is identity based, so a rename is *not* reported as a
        remove plus an add.

    dtype_changed: list of dict
        One entry per feature whose string dtype changed:
        ``{"name", "old_dtype", "new_dtype"}``.

    step_changes: list of dict
        Parameter / step-set differences between the two chains.
    """

    added: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    renamed: List[Dict] = field(default_factory=list)
    dtype_changed: List[Dict] = field(default_factory=list)
    step_changes: List[Dict] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (
            self.added
            or self.removed
            or self.renamed
            or self.dtype_changed
            or self.step_changes
        )

    def to_dict(self) -> Dict:
        return {
            "added": self.added,
            "removed": self.removed,
            "renamed": self.renamed,
            "dtype_changed": self.dtype_changed,
            "step_changes": self.step_changes,
        }

    def __repr__(self):
        return (
            "SnapshotDiff("
            f"added={len(self.added)}, removed={len(self.removed)}, "
            f"renamed={len(self.renamed)}, dtype_changed={len(self.dtype_changed)}, "
            f"step_changes={len(self.step_changes)})"
        )


def _live_index(snapshot: FeatureSnapshot) -> Dict[str, Dict]:
    """Index the live features of a snapshot by their stable identity."""

    return {
        record["identity"]: record for record in snapshot.features if record["alive"]
    }


def diff_snapshots(
    old_snapshot: Union[FeatureSnapshot, Dict],
    new_snapshot: Union[FeatureSnapshot, Dict],
) -> SnapshotDiff:
    """Compare two snapshots and return their :class:`SnapshotDiff`.

    Snapshots may be passed as :class:`FeatureSnapshot` objects or as the
    plain dictionaries returned by ``create_feature_snapshot``.
    """

    old = (
        old_snapshot
        if isinstance(old_snapshot, FeatureSnapshot)
        else FeatureSnapshot(old_snapshot)
    )
    new = (
        new_snapshot
        if isinstance(new_snapshot, FeatureSnapshot)
        else FeatureSnapshot(new_snapshot)
    )

    old_index = _live_index(old)
    new_index = _live_index(new)

    old_identities = set(old_index)
    new_identities = set(new_index)

    added = [
        new_index[identity]["name"]
        for identity in new_index
        if identity in (new_identities - old_identities)
    ]
    removed = [
        old_index[identity]["name"]
        for identity in old_index
        if identity in (old_identities - new_identities)
    ]

    renamed = []
    dtype_changed = []
    for identity in sorted(old_identities & new_identities):
        old_record = old_index[identity]
        new_record = new_index[identity]
        if old_record["name"] != new_record["name"]:
            renamed.append(
                {
                    "identity": identity,
                    "old_name": old_record["name"],
                    "new_name": new_record["name"],
                }
            )
        if (
            old_record["dtype"] is not None
            and new_record["dtype"] is not None
            and old_record["dtype"] != new_record["dtype"]
        ):
            dtype_changed.append(
                {
                    "name": new_record["name"],
                    "old_dtype": old_record["dtype"],
                    "new_dtype": new_record["dtype"],
                }
            )

    step_changes = _diff_steps(old, new)

    return SnapshotDiff(
        added=added,
        removed=removed,
        renamed=renamed,
        dtype_changed=dtype_changed,
        step_changes=step_changes,
    )


def _diff_steps(old: FeatureSnapshot, new: FeatureSnapshot) -> List[Dict]:
    changes: List[Dict] = []
    old_steps = {step["name"]: step for step in old.steps}
    new_steps = {step["name"]: step for step in new.steps}

    for name in new_steps.keys() - old_steps.keys():
        changes.append({"step": name, "change": "added"})
    for name in old_steps.keys() - new_steps.keys():
        changes.append({"step": name, "change": "removed"})

    for name in old_steps.keys() & new_steps.keys():
        old_step = old_steps[name]
        new_step = new_steps[name]
        if old_step["class_path"] != new_step["class_path"]:
            changes.append(
                {
                    "step": name,
                    "change": "class_changed",
                    "old_class": old_step["class_path"],
                    "new_class": new_step["class_path"],
                }
            )
        elif old_step["params"] != new_step["params"]:
            changed_params = {
                key: {
                    "old": old_step["params"].get(key),
                    "new": new_step["params"].get(key),
                }
                for key in set(old_step["params"]) | set(new_step["params"])
                if old_step["params"].get(key) != new_step["params"].get(key)
            }
            changes.append(
                {"step": name, "change": "params_changed", "params": changed_params}
            )
    return changes
