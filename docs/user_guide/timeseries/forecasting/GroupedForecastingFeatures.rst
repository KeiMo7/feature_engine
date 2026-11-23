.. _grouped_forecast_features:

Grouped (entity-aware) features
===============================

The forecasting features described so far assume that the dataframe contains a
single time series. Many real datasets, however, hold several independent series
in the same table: one row per store, sensor, customer or business line and
timestamp. Computing lags or rolling windows over such a table without grouping
would let one entity's values leak into another entity's history.

Feature-engine provides three entity-aware transformers:

- :class:`GroupedLagFeatures`
- :class:`GroupedWindowFeatures`
- :class:`GroupedExpandingWindowFeatures`

They mirror the API of their non-grouped counterparts and add the entity,
timestamp and forecasting controls described below. The non-grouped
transformers are unchanged, so existing pipelines keep their behaviour.

Grouping by entity
------------------

Pass the entity key column or columns in ``entity_variables`` and the time
column in ``timestamp``:

.. code:: python

    from feature_engine.timeseries.forecasting import GroupedWindowFeatures

    tr = GroupedWindowFeatures(
        entity_variables=["region", "store"],
        timestamp="date",
        window="7D",
        functions=["mean", "sum"],
    )
    Xt = tr.fit_transform(X)

Every lag and every window is computed within one entity. Entity codes are
assigned in order of first appearance, and each entity is sorted by its own
timestamps. The output keeps the original row order, the entity keys, the
timestamp and the original index, so it can be aligned with the target without
extra bookkeeping.

If an entity is absent from the training data and appears during ``transform``,
the transformer raises unless a cold-start policy is configured:

.. code:: python

    tr = GroupedWindowFeatures(
        entity_variables="store",
        timestamp="date",
        window=3,
        cold_start="nan",          # or "fill" with cold_start_value, or a number
    )

The default (``cold_start=None``) refuses unseen entities instead of silently
emitting zeros.

Time-based windows
------------------

An integer ``window`` counts observations. An offset such as ``"2D"`` or
``"45min"`` measures real time, so gaps and irregular sampling are respected.
When different entities are sampled at different frequencies, each entity is
aligned on its own timestamps; no resampling to a common grid is performed.

.. code:: python

    # mean of the observations of the same store in the 2 days before the row
    tr = GroupedWindowFeatures(
        entity_variables="store", timestamp="date", window="2D"
    )

A time window is the real-time interval between the target row and its
forecast cutoff. For a zero horizon it is ``[t - window, t)``: the lower bound
is inclusive and the current row is excluded. A gap larger than the window
therefore leaves it empty instead of pulling in a distant observation, while a
``"7D"`` window over daily data contains the previous seven observations. This
is the same convention as ``rolling(window).agg(...).shift(1)`` in the
non-grouped transformers, adapted to irregular timestamps.

Forecast horizon
----------------

``horizon`` is the forecast span: a feature for a row may only use observations
that are at least ``horizon`` older. It accepts rows (integer) or real time
(offset). An integer horizon counts prior observations of the same entity
(``0`` and ``1`` both mean "use the immediately preceding observation",
``2`` also skips it). An offset horizon uses the newest observation at or
before ``timestamp - horizon``. The current row and any row sharing its
timestamp are never part of its own history.

.. code:: python

    # features for the target at time t may only use data up to t - 1 day
    tr = GroupedWindowFeatures(
        entity_variables="store",
        timestamp="date",
        window="7D",
        horizon="1D",
    )

Duplicate timestamps
--------------------

Repeated ``(entity, timestamp)`` pairs are reported by default, including the
conflicting entity keys, time range and original row positions. Use
``duplicates="first"`` or ``duplicates="last"`` to keep one row, or
``duplicates="keep"`` to keep every row. Rows sharing a timestamp never enter
each other's history.

Minimum history
---------------

``min_periods`` controls how many observations a window needs before a value is
emitted. It defaults to the integer window size, or to 1 for time windows,
matching pandas. Entities with a shorter history get NaN instead of a
misleading statistic.

Feature names
-------------

Names are stable and traceable:

- ``{variable}_lag_{period}``
- ``{variable}_lag_{freq}``
- ``{variable}_window_{window}_{function}``
- ``{variable}_expanding_{function}``

Missing values, pipelines and ``transform_x_y`` work exactly as in the
non-grouped transformers.
