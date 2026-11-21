"""Build feature snapshots by tracing names, lineage and rename events.

A snapshot describes *structure only*: feature names, types, the step that
produces each feature, the input columns that step depended on and the full
parameter set of every step. No row of data is ever copied into a snapshot.

Feature identity
----------------
Every feature carries an ``identity`` string that is stable across renames.
Root columns are identified by their pipeline-input name (``root::<name>``).
A feature created by a step is identified by its birth step's class and the
*root* ancestor columns that fed it (``derive::<class>::<roots>``), so a
renamed created feature keeps the same identity even though its name changes.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Union

import pandas as pd

from feature_engine import __version__ as _fe_version
from feature_engine.snapshot._encoding import encode_dtype, encode_value
from feature_engine.snapshot._exceptions import FeatureNameConflictError

SNAPSHOT_FORMAT_VERSION = "1.0"

# Actions recorded in a feature's history.
_CREATED = "created"
_PASSTHROUGH = "passthrough"
_RENAMED = "renamed"
_DROPPED = "dropped"


def _step_class_path(transformer) -> str:
    return f"{transformer.__class__.__module__}.{transformer.__class__.__qualname__}"


def _short_class_path(class_path: str) -> str:
    """Last two components of the class path, enough to name the operation."""

    parts = class_path.split(".")
    return ".".join(parts[-2:])


def _declared_renames(transformer, explicit_renames: Optional[Dict[str, str]]):
    """Return the rename mapping declared for this step.

    Two declaration mechanisms coexist, and both are optional:

    - a ``renames`` mapping handed to ``create_feature_snapshot`` keyed by step
      name;
    - a ``feature_renames_`` fitted attribute on the transformer itself (this is
      the hook third-party transformers can implement).
    """

    renames = {}
    attr = getattr(transformer, "feature_renames_", None)
    if attr:
        renames.update(attr)
    if explicit_renames:
        renames.update(explicit_renames)
    return renames


def _infer_renames(input_cols: List, output_cols: List, renames: Dict[str, str]):
    """Validate declared renames and infer obvious 1:1 renames.

    A rename is any input column that is absent from the output while a
    matching declared target name appears in the output. When nothing is
    declared and exactly one input disappeared while exactly one new column
    appeared, the pair is treated as a rename.
    """

    output_set = set(output_cols)
    input_set = set(input_cols)
    inferred = {}

    for old_name, new_name in renames.items():
        if old_name not in input_set:
            raise ValueError(
                f"Declared rename source '{old_name}' is not an input column of "
                "the step. Check the `renames` mapping."
            )
        if new_name not in output_set:
            raise ValueError(
                f"Declared rename target '{new_name}' is not an output column of "
                "the step. Check the `renames` mapping."
            )
        inferred[old_name] = new_name

    disappeared = [c for c in input_cols if c not in output_set]
    appeared = [c for c in output_cols if c not in input_set]
    if not inferred and len(disappeared) == 1 and len(appeared) == 1:
        # Fallback with no explicit declaration: if exactly one column
        # disappeared and exactly one appeared, treat them as a rename. This
        # only fires in the strict 1:1 case; a same-name recreation is excluded
        # here and caught by the conflict detector. To force an explicit
        # interpretation, pass the mapping via RenameFeatures / `renames=`.
        if disappeared[0] != appeared[0]:
            inferred[disappeared[0]] = appeared[0]

    return inferred


def _recreated_columns(
    input_cols: List,
    output_cols: List,
    renamed_sources: set,
    renamed_targets: set,
) -> List:
    """Return shared-name columns that were dropped and recreated in place.

    The genuine survivors are the longest input-order-preserving subsequence
    of shared column names found in the output. Any shared name that is not
    part of that subsequence changed position and therefore corresponds to a
    column that was removed and re-created under the same name.
    """

    shared = [
        col
        for col in input_cols
        if col in set(output_cols)
        and col not in renamed_sources
        and col not in renamed_targets
    ]
    if not shared:
        return []

    output_index = {name: i for i, name in enumerate(output_cols)}
    positions = [output_index[col] for col in shared]

    # Longest non-decreasing subsequence (positions are unique so strictly
    # increasing): these shared columns kept their input relative order.
    n = len(positions)
    longest = [1] * n
    predecessor = [-1] * n
    for i in range(n):
        for j in range(i):
            if positions[j] < positions[i] and longest[j] + 1 > longest[i]:
                longest[i] = longest[j] + 1
                predecessor[i] = j

    end = max(range(n), key=lambda i: longest[i])
    survivor_indices = set()
    while end != -1:
        survivor_indices.add(end)
        end = predecessor[end]

    return [shared[i] for i in range(n) if i not in survivor_indices]


def build_snapshot(
    pipeline,
    X: Optional[pd.DataFrame] = None,
    *,
    renames: Optional[Dict[str, Dict[str, str]]] = None,
    name: Optional[str] = None,
    include_dtypes: bool = True,
    metadata_only: bool = False,
) -> Dict:
    """Trace the fitted pipeline and return a plain-dict feature snapshot.

    When ``X`` is provided the chain is executed once so that dtypes and
    exact per-step lineage can be observed. When ``X`` is ``None`` only
    fitted metadata (``get_feature_names_out``) is used, so no data is
    touched at all.
    """

    renames = renames or {}

    if X is None:
        metadata_only = True
        include_dtypes = False

    check_pipeline_is_fitted(pipeline)

    steps_payload = []
    # uid -> feature record
    features: Dict[str, Dict] = {}
    # current output column name -> uid
    name_to_uid: Dict[str, str] = {}
    # current output column name -> step that produced it (None = pipeline input)
    name_producer: Dict[str, Union[str, None]] = {}
    # current output column name -> sorted tuple of root input names feeding it
    name_roots: Dict[str, tuple] = {}
    # identity string -> number of features ever assigned it in this snapshot
    identity_counts: Dict[str, int] = {}
    uid_counter = [0]

    def new_uid() -> str:
        uid = f"f{uid_counter[0]}"
        uid_counter[0] += 1
        return uid

    # Seed the input columns as root features.
    if metadata_only:
        input_names = list(pipeline.feature_names_in_)
        input_dtypes = {}
    else:
        input_names = X.columns.tolist()
        input_dtypes = {c: X[c].dtype for c in input_names}

    for col in input_names:
        uid = new_uid()
        identity = f"root::{col}"
        identity_counts[identity] = 1
        record = {
            "uid": uid,
            "identity": identity,
            "name": col,
            "birth_step": None,
            "dtype": encode_dtype(input_dtypes[col]) if include_dtypes else None,
            "dependencies": [],
            "history": [{"step": None, "action": _CREATED, "name": col}],
            "alive": True,
        }
        features[uid] = record
        name_to_uid[col] = uid
        name_producer[col] = None
        name_roots[col] = (col,)

    current = X
    # When the last step is itself a transformer it is stored as the pipeline's
    # "final estimator" and therefore skipped by _iter(with_final=False). It is
    # still part of the feature chain, so iterate with the final estimator and
    # skip it below only when it is a genuine predictor.
    last_is_transformer = pipeline._final_estimator == "passthrough" or hasattr(
        pipeline._final_estimator, "transform"
    )
    for step_index, step_name, transformer in pipeline._iter(
        with_final=last_is_transformer, filter_passthrough=False
    ):
        if transformer is None or transformer == "passthrough":
            steps_payload.append(
                {
                    "index": step_index,
                    "name": step_name,
                    "class_path": "passthrough",
                    "params": {},
                    "library_version": _fe_version,
                    "input_features": list(name_to_uid.keys()),
                    "output_features": list(name_to_uid.keys()),
                    "renames": {},
                }
            )
            continue

        input_cols = list(name_to_uid.keys())
        step_renames = _declared_renames(transformer, renames.get(step_name))

        if metadata_only:
            output_cols = list(transformer.get_feature_names_out())
            output_dtypes = {}
            current = None
        else:
            current = transformer.transform(current)
            output_cols = current.columns.tolist()

        # A single step emitting the same name more than once is invalid and
        # must be rejected before any per-column access (duplicate labels make
        # ``current[col]`` return a DataFrame rather than a Series).
        output_set = set(output_cols)
        if len(output_set) != len(output_cols):
            seen = set()
            duplicate = next(c for c in output_cols if c in seen or seen.add(c))
            raise FeatureNameConflictError(duplicate, step_name)

        if metadata_only:
            output_dtypes = {}
        else:
            output_dtypes = {c: current[c].dtype for c in output_cols}

        rename_map = _infer_renames(input_cols, output_cols, step_renames)

        input_set = set(input_cols)
        renamed_sources = set(rename_map.keys())
        renamed_targets = set(rename_map.values())

        # ---- conflict detection -------------------------------------------
        # (b) an input column that disappeared but its name reappears in the
        # output without being declared as a rename target: the original was
        # dropped and a different column reuses its name. pandas would replace
        # it silently, so report both producers. Name sets alone cannot tell a
        # passthrough from a same-name recreation, so we use column order: the
        # longest order-preserving subsequence of input names inside the output
        # are the genuine survivors; any shared-name column outside it was
        # recreated at a new position.
        recreated = _recreated_columns(
            input_cols, output_cols, renamed_sources, renamed_targets
        )
        for col in recreated:
            producer = name_producer.get(col)
            raise FeatureNameConflictError(
                col,
                producer if producer is not None else "input",
                step_name,
            )

        new_cols = [
            c for c in output_cols if c not in input_set and c not in renamed_targets
        ]

        # (c) a step emitting a brand-new column whose name collides with a
        # column that already exists and that was not declared as a rename.
        for col in new_cols:
            if col in name_producer:
                producer = name_producer[col]
                raise FeatureNameConflictError(
                    col,
                    producer if producer is not None else "input",
                    step_name,
                )

        # ---- renames: same uid and identity, only the name changes --------
        inverse_renames = {new: old for old, new in rename_map.items()}
        for old_name, new_name in rename_map.items():
            uid = name_to_uid[old_name]
            record = features[uid]
            # The identity is lineage based and therefore must NOT change:
            # a rename is the same feature under a new name.
            record["name"] = new_name
            record["history"].append(
                {
                    "step": step_name,
                    "action": _RENAMED,
                    "from": old_name,
                    "name": new_name,
                }
            )
            if include_dtypes and new_name in output_dtypes:
                record["dtype"] = encode_dtype(output_dtypes[new_name])

        # ---- dropped columns ----------------------------------------------
        renamed_sources = set(rename_map.keys())
        for col in input_cols:
            if col in output_set or col in renamed_sources:
                continue
            uid = name_to_uid[col]
            record = features[uid]
            record["alive"] = False
            record["history"].append(
                {"step": step_name, "action": _DROPPED, "name": col}
            )

        # ---- surviving columns --------------------------------------------
        surviving = [
            c for c in output_cols if c in input_set and c not in rename_map.values()
        ]
        for col in surviving:
            record = features[name_to_uid[col]]
            record["history"].append(
                {"step": step_name, "action": _PASSTHROUGH, "name": col}
            )
            if include_dtypes and col in output_dtypes:
                record["dtype"] = encode_dtype(output_dtypes[col])

        # ---- brand new columns --------------------------------------------
        step_class_path = _step_class_path(transformer)
        short_class = _short_class_path(step_class_path)
        created_uids = {}
        for col in new_cols:
            uid = new_uid()
            dependency_names = _step_dependencies(transformer, input_cols)
            # Roots actually feeding the new column: union of the roots of
            # every dependency that is currently alive.
            root_set = set()
            for dep in dependency_names:
                if dep in name_roots:
                    root_set.update(name_roots[dep])
            roots = tuple(sorted(root_set))
            base_identity = f"derive::{short_class}::{'+'.join(roots)}"
            # Two distinct steps of the same class over the same root columns
            # are different features: disambiguate with a stable ordinal.
            ordinal = identity_counts.get(base_identity, 0)
            identity_counts[base_identity] = ordinal + 1
            identity = base_identity if ordinal == 0 else f"{base_identity}#{ordinal}"
            record = {
                "uid": uid,
                "identity": identity,
                "name": col,
                "birth_step": step_name,
                "dtype": encode_dtype(output_dtypes[col]) if include_dtypes else None,
                "dependencies": dependency_names,
                "history": [{"step": step_name, "action": _CREATED, "name": col}],
                "alive": True,
            }
            features[uid] = record
            created_uids[col] = uid

        # ---- rebuild the column maps in physical output order -------------
        # pandas keeps a renamed column in its original position, so the maps
        # must follow the output column order rather than insertion order.
        next_name_to_uid = {}
        next_name_producer = {}
        next_name_roots = {}
        for col in output_cols:
            if col in inverse_renames:
                source = inverse_renames[col]
                uid = name_to_uid[source]
                roots = name_roots[source]
                producer = step_name
            elif col in created_uids:
                uid = created_uids[col]
                roots = tuple(
                    sorted(
                        {
                            root
                            for dep in features[uid]["dependencies"]
                            if dep in name_roots
                            for root in name_roots[dep]
                        }
                    )
                )
                producer = step_name
            else:
                uid = name_to_uid[col]
                roots = name_roots[col]
                producer = name_producer[col]
            next_name_to_uid[col] = uid
            next_name_producer[col] = producer
            next_name_roots[col] = roots
        name_to_uid = next_name_to_uid
        name_producer = next_name_producer
        name_roots = next_name_roots

        steps_payload.append(
            {
                "index": step_index,
                "name": step_name,
                "class_path": step_class_path,
                "params": {
                    key: encode_value(value)
                    for key, value in transformer.get_params(deep=False).items()
                },
                "library_version": _fe_version,
                "input_features": input_cols,
                "output_features": output_cols,
                "renames": rename_map,
            }
        )

    final_output_cols = list(name_to_uid.keys())
    final_features = [features[name_to_uid[col]] for col in final_output_cols]
    alive_set = set(uid for uid in name_to_uid.values())
    dropped_uids = sorted(
        (uid for uid in features if uid not in alive_set),
        key=lambda uid: int(uid[1:]),
    )
    final_features.extend(features[uid] for uid in dropped_uids)

    snapshot = {
        "format_version": SNAPSHOT_FORMAT_VERSION,
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "library": {
            "name": "feature-engine",
            "version": _fe_version,
        },
        "environment": {
            "python": _python_version(),
            "pandas": pd.__version__,
        },
        "input_features": input_names,
        "steps": steps_payload,
        "features": final_features,
    }
    return snapshot


def _step_dependencies(transformer, input_cols: List[str]) -> List[str]:
    """Best-effort list of input columns a step actually consumed."""

    variables_ = getattr(transformer, "variables_", None)
    if variables_ is not None:
        return list(variables_)
    return list(input_cols)


def check_pipeline_is_fitted(pipeline):
    for _, _, transformer in pipeline._iter(with_final=True, filter_passthrough=False):
        if transformer is None or transformer == "passthrough":
            continue
        if not hasattr(transformer, "fit"):
            continue
        try:
            from sklearn.utils.validation import check_is_fitted

            check_is_fitted(transformer)
        except Exception as exc:
            raise ValueError(
                "Pipeline must be fitted before a feature snapshot can be " "created."
            ) from exc


def _python_version() -> str:
    import sys

    return ".".join(str(part) for part in sys.version_info[:3])
