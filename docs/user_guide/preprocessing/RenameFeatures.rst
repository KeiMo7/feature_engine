.. _rename_features:

.. currentmodule:: feature_engine.preprocessing

RenameFeatures
==============

:class:`RenameFeatures()` renames variables according to a user-supplied
mapping. It leaves the values and dtypes of every column untouched and only
changes the column labels.

Using :class:`RenameFeatures()` instead of an ad-hoc
``pandas.DataFrame.rename`` call has one important benefit for feature
lineage: the transformer stores the mapping in the fitted attribute
``feature_renames_``. When the pipeline is snapshotted (see
:ref:`feature_snapshots`), a renamed column is therefore recognised as the
*same* feature under a new name rather than as a removed feature plus a new
feature.

Example
-------

We start with the imports and a toy dataframe:

.. code:: python

    import pandas as pd
    from feature_engine.preprocessing import RenameFeatures

    X = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6]})

We rename ``x1`` to ``age``:

.. code:: python

    transformer = RenameFeatures(rename_dict={"x1": "age"})
    transformer.fit_transform(X)

In the output, the column is renamed but its values are unchanged:

.. code:: python

       age  x2
    0    1   4
    1    2   5
    2    3   6

The validated mapping is available after fitting:

.. code:: python

    transformer.feature_renames_

.. code:: python

    {'x1': 'age'}

The transformer raises an error if a rename source is not a column of the
training dataframe, if two columns are mapped to the same new name, or if a
rename target already exists as another column.
