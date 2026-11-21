.. _feature_snapshots:

.. currentmodule:: feature_engine.pipeline

Feature snapshots, diffs and rollback
=====================================

A feature snapshot is a versioned, JSON-serialisable description of the feature
set produced by a :class:`Pipeline`. It records the *structure* of the feature
set — names, dtypes, the step that produced each feature, the input columns
that step depended on, the full parameter set of every step and the library
versions in use — but it never contains rows of data.

Snapshots are opt-in. If you never call the snapshot methods, the pipeline
behaves exactly as before and no extra work is performed.

Creating a snapshot
-------------------

The pipeline must be fitted first. Pass the training data to trace the chain
through its actual outputs and record dtypes:

.. code:: python

    import pandas as pd
    from feature_engine.creation import MathFeatures
    from feature_engine.pipeline import Pipeline
    from feature_engine.preprocessing import RenameFeatures
    from feature_engine.selection import DropFeatures

    X = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [5.0, 6.0, 7.0, 8.0],
            "x3": [9.0, 10.0, 11.0, 12.0],
        }
    )

    pipe = Pipeline(
        [
            ("math", MathFeatures(variables=["x1", "x2"], func=["sum", "mean"])),
            ("rename", RenameFeatures(rename_dict={"x3": "x3_renamed"})),
            ("drop", DropFeatures(features_to_drop=["x2"])),
        ]
    ).fit(X)

    snapshot = pipe.create_feature_snapshot(X, name="feature-set-v1")

Persist it to disk and reload it later:

.. code:: python

    path = snapshot.to_json_file("snapshot_v1.json")

    from feature_engine.snapshot import FeatureSnapshot

    reloaded = FeatureSnapshot.from_json_file(path)

If you cannot or do not want to run data through the chain, call
``create_feature_snapshot()`` with no arguments. The snapshot is then built
from the fitted ``get_feature_names_out()`` metadata only: no data is touched,
but dtypes are not recorded.

Reading a snapshot
------------------

``snapshot.feature_names`` lists the features alive at the end of the chain,
in output order. ``snapshot.lineage(name)`` returns the full trace of one
feature — the step that created it, its input dependencies and every event in
its history:

.. code:: python

    snapshot.lineage("mean_x1_x2")

Each event in the history is one of ``created``, ``passthrough``, ``renamed``
or ``dropped``, tagged with the step that produced it.

Diffing two snapshots
---------------------

``diff_snapshots`` reports four feature-level categories separately — added,
removed, renamed and dtype changed — plus the parameter changes between the
two chains:

.. code:: python

    from feature_engine.snapshot import diff_snapshots

    diff = diff_snapshots(old_snapshot, new_snapshot)
    diff.added
    diff.removed
    diff.renamed
    diff.dtype_changed
    diff.step_changes

You can also diff a stored snapshot against the current fitted pipeline:

.. code:: python

    diff = pipe.diff_feature_snapshots(old_snapshot)

Stable feature identity
-----------------------

A rename is recognised as the *same* feature under a new name, not as a remove
plus an add. Every feature carries an ``identity`` that follows its lineage:
root input columns use ``root::<input-name>``, while a feature derived by a
step is keyed by the transformer class together with the root input columns
feeding it (``derive::<class>::<roots>``). Because the identity never embeds
the current feature name, renaming a column leaves it unchanged and the diff
reports a single ``renamed`` entry.

The built-in :class:`~feature_engine.preprocessing.RenameFeatures`
transformer records its mapping in the fitted attribute
``feature_renames_``, which the snapshot machinery reads automatically. For
third-party transformers that rename columns, pass an explicit mapping keyed
by step name:

.. code:: python

    snapshot = pipe.create_feature_snapshot(
        X, renames={"my_step": {"old_name": "new_name"}}
    )

Rolling back
------------

``rollback`` rebuilds the unfitted transform chain stored in a snapshot. When
you also pass data, the rebuilt pipeline is fitted and its output feature set
is verified against the snapshot:

.. code:: python

    from feature_engine.snapshot import rollback

    restored_pipeline = rollback("snapshot_v1.json", X, verify=True)
    restored_pipeline.transform(X).columns

Rollback restores the chain and its parameters, not fitted state: the restored
pipeline is re-fit on the data you provide, so it regenerates the same
*feature set*. If the snapshot cannot be restored — for example because a
transformer class no longer exists in the installed library or its constructor
rejects the stored parameters — a
:class:`~feature_engine.snapshot.SnapshotVersionError` is raised with concrete
migration advice.

Name conflicts
--------------

Two conflict situations raise
:class:`~feature_engine.snapshot.FeatureNameConflictError`, which names the
feature and the two steps involved:

- a single step emits the same output column name more than once;
- a step creates a column whose name collides with a column that already
  exists, without declaring it as a rename (pandas would otherwise overwrite
  the existing column silently).

A transformer that changes the values of a column in place while keeping its
name is treated as the same feature; name-based lineage cannot distinguish it
from a passthrough, and this is a documented limitation.

Corrupt snapshot files
----------------------

Loading a broken file raises
:class:`~feature_engine.snapshot.SnapshotCorruptionError` whose ``segment``
attribute pinpoints the problem: the file itself (invalid JSON, with line and
column), ``root`` (a missing top-level section), ``steps[i]``,
``features[i]`` or ``format_version``.

Known limitations
-----------------

- Snapshots describe structure and parameters, never data.
- In-place value changes that preserve the column name are treated as the same
  feature.
- When no rename is declared and a step drops exactly one column while adding
  exactly one differently named column, the pair is assumed to be a rename.
  Pass an explicit mapping (``RenameFeatures`` or ``renames=``) whenever the
  correct interpretation matters.
- Parameters that are not JSON portable (for example a lambda function) are
  flagged in the snapshot; use a top-level, importable function or a string
  alias instead so that the snapshot can be rolled back.
- A final *predictor* step is not part of the feature set and is therefore not
  recorded; only the transformation chain is snapshotted.
