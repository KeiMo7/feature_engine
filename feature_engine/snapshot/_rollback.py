"""Roll a pipeline back to the chain stored in a feature snapshot.

Rollback rebuilds the *unfitted* transform chain from the class paths and
parameters stored in the snapshot. Fitting the rebuilt pipeline on data
regenerates the same feature set; when the input data is also supplied the
rebuilt pipeline is fitted, run and its output feature set is verified
against the snapshot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Union
from pathlib import Path

import pandas as pd

from feature_engine.snapshot._encoding import _import_class_path, decode_value
from feature_engine.snapshot._exceptions import SnapshotVersionError
from feature_engine.snapshot._snapshot import FeatureSnapshot

if TYPE_CHECKING:
    from feature_engine.pipeline import Pipeline


def restore_pipeline(
    snapshot: Union[FeatureSnapshot, Dict, str, Path],
) -> Pipeline:
    """Rebuild the (unfitted) pipeline stored in a snapshot.

    Parameters
    ----------
    snapshot: FeatureSnapshot, dict, str or Path
        The snapshot to roll back to. A path is loaded from disk (with
        corruption checks), a dictionary is validated on construction.

    Returns
    -------
    pipeline: Pipeline
        A fresh, unfitted :class:`~feature_engine.pipeline.Pipeline` whose
        steps and parameters match the snapshot exactly.
    """

    snapshot = _coerce_snapshot(snapshot)
    from feature_engine.pipeline import Pipeline

    payload = snapshot.to_dict()
    installed_version = payload["library"]["version"]
    steps = []

    for step in payload["steps"]:
        if step["class_path"] == "passthrough":
            steps.append((step["name"], "passthrough"))
            continue
        try:
            transformer_class = _import_class_path(step["class_path"])
        except (ImportError, AttributeError) as exc:
            raise SnapshotVersionError(
                step_name=step["name"],
                class_path=step["class_path"],
                snapshot_version=step.get("library_version", installed_version),
                reason=(
                    "the transformer class cannot be imported from the installed "
                    f"library ({type(exc).__name__})"
                ),
                advice=(
                    "install the feature-engine version that ships this class "
                    f"(snapshot recorded {step.get('library_version')}), or apply "
                    "the project's migration guide and re-save the snapshot with "
                    "the replacement transformer"
                ),
            ) from exc

        try:
            params = {key: decode_value(value) for key, value in step["params"].items()}
            transformer = transformer_class(**params)
        except (TypeError, ValueError) as exc:
            raise SnapshotVersionError(
                step_name=step["name"],
                class_path=step["class_path"],
                snapshot_version=step.get("library_version", installed_version),
                reason=f"the current class rejects the stored parameters ({exc})",
                advice=(
                    "the transformer's constructor likely changed between versions; "
                    "map the stored parameters to the new constructor arguments and "
                    "re-save the snapshot, or pin the library version recorded in the "
                    "snapshot"
                ),
            ) from exc

        steps.append((step["name"], transformer))

    return Pipeline(steps)


def rollback(
    snapshot: Union[FeatureSnapshot, Dict, str, Path],
    X: Optional[pd.DataFrame] = None,
    y=None,
    *,
    verify: bool = True,
) -> Pipeline:
    """Restore the snapshot's chain and, when ``X`` is given, fit and verify it.

    Parameters
    ----------
    snapshot: FeatureSnapshot, dict, str or Path
        The snapshot to roll back to.

    X: pandas dataframe, default=None
        Data to fit the restored pipeline on. When provided, the pipeline's
        output feature set is compared against the snapshot.

    y: pandas Series, default=None
        Optional target forwarded to ``fit``.

    verify: bool, default=True
        When True (and ``X`` is provided) raise if the regenerated feature set
        does not match the snapshot's feature names and order exactly.

    Returns
    -------
    pipeline: Pipeline
        The restored pipeline, fitted when ``X`` was provided.
    """

    snapshot = _coerce_snapshot(snapshot)
    pipeline = restore_pipeline(snapshot)

    if X is None:
        return pipeline

    pipeline.fit(X, y)
    if verify:
        expected = snapshot.feature_names
        observed = pipeline.get_feature_names_out()
        if list(observed) != list(expected):
            raise AssertionError(
                "Rollback verification failed: the regenerated feature set does "
                f"not match the snapshot.\n  expected: {expected}\n  "
                f"observed: {list(observed)}"
            )
    return pipeline


def _coerce_snapshot(
    snapshot: Union[FeatureSnapshot, Dict, str, Path],
) -> FeatureSnapshot:
    if isinstance(snapshot, FeatureSnapshot):
        return snapshot
    if isinstance(snapshot, (str, Path)):
        return FeatureSnapshot.from_json_file(snapshot)
    return FeatureSnapshot(snapshot)
