"""The FeatureSnapshot object: in-memory wrapper plus file IO."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from feature_engine.snapshot._exceptions import SnapshotCorruptionError
from feature_engine.snapshot._tracking import SNAPSHOT_FORMAT_VERSION

_REQUIRED_TOP_LEVEL = (
    "format_version",
    "created_at",
    "library",
    "steps",
    "features",
    "input_features",
)
_REQUIRED_FEATURE_KEYS = (
    "uid",
    "identity",
    "name",
    "birth_step",
    "dtype",
    "dependencies",
    "history",
    "alive",
)
_REQUIRED_STEP_KEYS = (
    "index",
    "name",
    "class_path",
    "params",
    "input_features",
    "output_features",
)


class FeatureSnapshot:
    """A versioned, serialisable description of a pipeline's feature set.

    Holds structure and parameters only — never data rows.

    Parameters
    ----------
    payload : dict
        The plain dictionary produced by
        :meth:`~feature_engine.pipeline.Pipeline.create_feature_snapshot`.
    """

    def __init__(self, payload: Dict):
        self._validate(payload)
        self.payload = payload

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, payload: Dict) -> "FeatureSnapshot":
        """Build a snapshot from an already-loaded dictionary."""

        return cls(payload)

    @classmethod
    def from_json_file(cls, path: Union[str, Path]) -> "FeatureSnapshot":
        """Load a snapshot from a JSON file.

        Raises
        ------
        SnapshotCorruptionError
            If the file is not valid JSON or a required segment is missing
            or malformed. The error names the broken segment.
        """

        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SnapshotCorruptionError(
                "file", f"cannot read snapshot file '{path}': {exc}"
            ) from exc

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SnapshotCorruptionError(
                "file",
                f"invalid JSON at line {exc.lineno}, column {exc.colno} "
                f"(near: {exc.msg})",
            ) from exc

        return cls(payload)

    # ------------------------------------------------------------------ #
    # serialisation
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict:
        """Return the snapshot as a plain, JSON-serialisable dictionary."""

        return self.payload

    def to_json_file(self, path: Union[str, Path], indent: int = 2) -> Path:
        """Write the snapshot to a JSON file and return its path."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.payload, indent=indent, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    # ------------------------------------------------------------------ #
    # convenience views
    # ------------------------------------------------------------------ #
    @property
    def feature_names(self) -> List[str]:
        """Names of the features alive at the end of the chain."""

        return [f["name"] for f in self.payload["features"] if f["alive"]]

    @property
    def steps(self) -> List[Dict]:
        return self.payload["steps"]

    @property
    def features(self) -> List[Dict]:
        return self.payload["features"]

    def feature(self, name: str) -> Optional[Dict]:
        """Return the record of a feature by its current name."""

        for record in self.payload["features"]:
            if record["name"] == name and record["alive"]:
                return record
        return None

    def lineage(self, name: str) -> Dict:
        """Return the full trace (birth step, dependencies, history) of a feature."""

        record = self.feature(name)
        if record is None:
            raise KeyError(f"Feature '{name}' is not present in this snapshot.")
        return {
            "name": record["name"],
            "identity": record["identity"],
            "birth_step": record["birth_step"],
            "dependencies": record["dependencies"],
            "dtype": record["dtype"],
            "history": record["history"],
        }

    # ------------------------------------------------------------------ #
    # validation
    # ------------------------------------------------------------------ #
    def _validate(self, payload: Dict):
        if not isinstance(payload, dict):
            raise SnapshotCorruptionError(
                "root", f"snapshot must be a JSON object, got {type(payload).__name__}"
            )

        missing = [key for key in _REQUIRED_TOP_LEVEL if key not in payload]
        if missing:
            raise SnapshotCorruptionError(
                "root", f"missing required section(s): {missing}"
            )

        if payload["format_version"] not in _SUPPORTED_FORMAT_VERSIONS:
            raise SnapshotCorruptionError(
                "format_version",
                f"unsupported snapshot format version "
                f"'{payload['format_version']}', supported: "
                f"{sorted(_SUPPORTED_FORMAT_VERSIONS)}",
            )

        if not isinstance(payload["steps"], list):
            raise SnapshotCorruptionError("steps", "section must be a list")
        if not isinstance(payload["features"], list):
            raise SnapshotCorruptionError("features", "section must be a list")

        for index, step in enumerate(payload["steps"]):
            segment = f"steps[{index}]"
            if not isinstance(step, dict):
                raise SnapshotCorruptionError(segment, "step must be an object")
            step_missing = [k for k in _REQUIRED_STEP_KEYS if k not in step]
            if step_missing:
                raise SnapshotCorruptionError(segment, f"missing keys: {step_missing}")
            if not isinstance(step["params"], dict):
                raise SnapshotCorruptionError(
                    f"{segment}.params", "params must be an object"
                )

        seen_uids = set()
        seen_names = set()
        for index, feature in enumerate(payload["features"]):
            segment = f"features[{index}]"
            if not isinstance(feature, dict):
                raise SnapshotCorruptionError(segment, "feature must be an object")
            feature_missing = [k for k in _REQUIRED_FEATURE_KEYS if k not in feature]
            if feature_missing:
                raise SnapshotCorruptionError(
                    segment, f"missing keys: {feature_missing}"
                )
            if feature["uid"] in seen_uids:
                raise SnapshotCorruptionError(
                    segment, f"duplicate feature uid '{feature['uid']}'"
                )
            seen_uids.add(feature["uid"])
            if feature["alive"]:
                if feature["name"] in seen_names:
                    raise SnapshotCorruptionError(
                        segment,
                        f"duplicate live feature name '{feature['name']}'",
                    )
                seen_names.add(feature["name"])
            if not isinstance(feature["history"], list) or not feature["history"]:
                raise SnapshotCorruptionError(
                    f"{segment}.history",
                    "history must be a non-empty list of events",
                )

    def __repr__(self):
        n_alive = sum(1 for f in self.payload["features"] if f["alive"])
        n_steps = len(self.payload["steps"])
        return (
            f"FeatureSnapshot(name={self.payload.get('name')!r}, "
            f"steps={n_steps}, features={n_alive}, "
            f"format_version={self.payload['format_version']!r})"
        )


_SUPPORTED_FORMAT_VERSIONS = {SNAPSHOT_FORMAT_VERSION}
