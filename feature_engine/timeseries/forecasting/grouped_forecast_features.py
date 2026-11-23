"""Entity-aware (grouped) lag, rolling and expanding time-series features.

The transformers in this module extend Feature-engine's forecasting features to
tabular data that contains several independent time series in the same table
(for example, one row per store, sensor or business line and timestamp).
They compute every lag and window **within an entity**, order observations by a
real timestamp, apply a configurable forecast horizon, and restore the original
row order so that entity keys, timestamps and feature values stay aligned.

The existing non-grouped transformers (``LagFeatures``, ``WindowFeatures`` and
``ExpandingWindowFeatures``) are not modified: they keep their public API and
their results.
"""

from __future__ import annotations

import numbers
from typing import Hashable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.indexers import BaseIndexer
from pandas.api.types import is_datetime64_any_dtype
from sklearn.utils.validation import check_is_fitted

from feature_engine._check_init_parameters.check_init_input_params import (
    _check_return_empty_is_bool,
)
from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine.dataframe_checks import (
    _check_X_matches_training_df,
    check_X,
)
from feature_engine.timeseries.forecasting.base_forecast_transformers import (
    BaseForecastTransformer,
)
from feature_engine.variable_handling import check_numerical_variables

Variables = Union[None, int, str, List[Union[str, int]]]

_ALLOWED_COLD_START = (None, "raise", "nan", "fill")
_ALLOWED_DUPLICATES = ("raise", "keep", "first", "last")


def _is_int(value) -> bool:
    """True for integers but not for booleans."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _is_offset(value) -> bool:
    return isinstance(value, (str, pd.Timedelta))


def _to_timedelta(value, name: str) -> pd.Timedelta:
    """Parse a fixed-duration offset and raise a clear error otherwise."""
    try:
        delta = pd.to_timedelta(value)
    except (TypeError, ValueError) as err:
        raise ValueError(
            f"`{name}` must be a fixed time offset such as '1D' or '2h'. "
            "Month and year offsets are not supported. "
            f"Got {value!r} instead."
        ) from err
    if pd.isna(delta):
        raise ValueError(f"`{name}` must be a valid time offset. Got {value!r}.")
    return delta


def _offset_label(value) -> Union[str, int]:
    """Return a stable, traceable label for a window or lag specification."""
    if isinstance(value, pd.Timedelta):
        try:
            return pd.tseries.frequencies.to_offset(value).freqstr
        except (ValueError, TypeError):
            return str(value)
    return value


def _as_list(value) -> list:
    return value if isinstance(value, list) else [value]


class _GroupedWindowIndexer(BaseIndexer):
    """Rolling indexer with explicit per-row window bounds.

    ``starts`` and ``ends`` are the half-open positional bounds of the window
    ending at each row. Bounds never cross an entity block.
    """

    def __init__(self, starts: np.ndarray, ends: np.ndarray) -> None:
        self.starts = np.asarray(starts, dtype=np.int64)
        self.ends = np.asarray(ends, dtype=np.int64)
        span = int(np.max(self.ends - self.starts)) if len(self.starts) else 1
        super().__init__(window_size=max(1, span))

    def get_window_bounds(
        self,
        num_values: int,
        min_periods: Optional[int],
        center: bool,
        closed: Optional[str],
        step: Optional[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.starts, self.ends


class BaseGroupedForecastTransformer(BaseForecastTransformer):
    """Shared machinery for entity-aware forecasting transformers.

    The class validates entity keys and timestamps, orders observations within
    each entity, handles duplicate timestamps according to ``duplicates``, and
    restores the original row order. Subclasses implement the feature
    computation itself.

    Parameters
    ----------
    variables: str, int, or list of strings or integers, default=None.
        The numerical variables used to create the new features. Entity and
        timestamp columns are always excluded.

    entity_variables: str, int, or list, default=None.
        Column or columns that identify the entity (business line, store,
        sensor, ...). Required to fit the transformer.

    timestamp: str or int, default=None.
        Column with the observation time. Values are parsed with
        ``pandas.to_datetime`` when they are not already datetime. Required to
        fit the transformer.

    return_empty: bool, default=False.
        Whether to allow an empty list of numerical variables.

    horizon: int, str, or Timedelta, default=0.
        Forecast span. A feature for a row may only use observations that are
        at least ``horizon`` away from it: with an integer, the newest usable
        observation is ``max(horizon, 1)`` rows back within the entity; with
        an offset, it is the last observation at or before
        ``timestamp - horizon``. The current row and all rows sharing its
        timestamp are always excluded from its own history.

    cold_start: None, "raise", "nan", "fill", or a number, default=None.
        Policy for entities that appear in ``transform`` but were absent from
        ``fit``. ``None`` and ``"raise"`` raise an error; ``"nan"`` produces
        missing feature values; ``"fill"`` fills them with
        ``cold_start_value``; a number is used directly as the fill value.

    cold_start_value: number, default=None.
        Value used when ``cold_start="fill"``.

    duplicates: str, default="raise".
        How to treat repeated ``(entity, timestamp)`` pairs. ``"raise"`` reports
        the conflicting rows, ``"first"`` / ``"last"`` keep one row and drop the
        other duplicates, and ``"keep"`` keeps every row. Rows sharing a
        timestamp never enter each other's history.

    date_format: str, default=None.
        Format passed to ``pandas.to_datetime`` for non-datetime timestamps.

    missing_values: str, default="raise".
        Whether to raise or ignore missing and infinite values in ``variables``.

    drop_original: bool, default=False.
        Whether to drop the original variables from the output.

    drop_na: bool, default=False.
        Whether to drop rows with missing values in the created features.

    Attributes
    ----------
    variables_:
        The numerical variables used to create the features.

    entity_values_:
        The entity keys observed during ``fit``.

    time_range_:
        Tuple with the first and last timestamp observed during ``fit``.

    feature_names_in_:
        List with the names of features seen during ``fit``.

    n_features_in_:
        The number of features in the train set used in fit.
    """

    def __init__(
        self,
        variables: Variables = None,
        entity_variables: Variables = None,
        timestamp: Union[str, int, None] = None,
        return_empty: bool = False,
        horizon: Union[int, str, pd.Timedelta] = 0,
        cold_start: Union[None, str, numbers.Number] = None,
        cold_start_value: Optional[numbers.Number] = None,
        duplicates: str = "raise",
        date_format: Optional[str] = None,
        missing_values: str = "raise",
        drop_original: bool = False,
        drop_na: bool = False,
    ) -> None:
        super().__init__(
            variables, return_empty, missing_values, drop_original, drop_na
        )

        self.entity_variables = _check_variables_input_value(entity_variables)

        if timestamp is not None and not isinstance(timestamp, (str, int)):
            raise ValueError(
                "`timestamp` must be the name of a single column. "
                f"Got {timestamp!r} instead."
            )
        self.timestamp = timestamp

        if _is_int(horizon):
            if int(horizon) < 0:
                raise ValueError(
                    "`horizon` must be a non-negative integer or a time offset. "
                    f"Got {horizon!r} instead."
                )
        elif _is_offset(horizon):
            if _to_timedelta(horizon, "horizon") < pd.Timedelta(0):
                raise ValueError(
                    "`horizon` must be a non-negative integer or a time offset. "
                    f"Got {horizon!r} instead."
                )
        else:
            raise ValueError(
                "`horizon` must be a non-negative integer or a time offset such "
                f"as '1D'. Got {horizon!r} instead."
            )
        self.horizon = horizon

        if duplicates not in _ALLOWED_DUPLICATES:
            raise ValueError(
                f"`duplicates` takes only values {_ALLOWED_DUPLICATES}. "
                f"Got {duplicates!r} instead."
            )
        self.duplicates = duplicates

        numeric_cold_start = isinstance(cold_start, numbers.Number) and not isinstance(
            cold_start, bool
        )
        if cold_start not in _ALLOWED_COLD_START and not numeric_cold_start:
            raise ValueError(
                "`cold_start` takes values None, 'raise', 'nan', 'fill', or a "
                f"number used as the fill value. Got {cold_start!r} instead."
            )
        if cold_start == "fill" and cold_start_value is None:
            raise ValueError(
                "When `cold_start='fill'` you must set `cold_start_value` to the "
                "value used for unseen entities."
            )
        if cold_start_value is not None and not isinstance(
            cold_start_value, numbers.Number
        ):
            raise ValueError(
                "`cold_start_value` must be a number. "
                f"Got {cold_start_value!r} instead."
            )
        self.cold_start = cold_start
        self.cold_start_value = cold_start_value

        if date_format is not None and not isinstance(date_format, str):
            raise ValueError(
                f"`date_format` must be a string or None. Got {date_format!r}."
            )
        self.date_format = date_format

        _check_return_empty_is_bool(return_empty)

    # ------------------------------------------------------------------
    # validation helpers
    # ------------------------------------------------------------------
    def _entity_columns_list(self) -> List[Union[str, int]]:
        if self.entity_variables is None:
            raise ValueError(
                "`entity_variables` is required to fit a grouped forecasting "
                "transformer."
            )
        if isinstance(self.entity_variables, (str, int)):
            return [self.entity_variables]
        return list(self.entity_variables)

    def _validate_columns(self, X: pd.DataFrame) -> None:
        if self.timestamp is None:
            raise ValueError(
                "`timestamp` is required to fit a grouped forecasting transformer."
            )
        entity_columns = self._entity_columns_list()
        if len(entity_columns) == 0:
            raise ValueError("`entity_variables` cannot be empty.")
        if len(entity_columns) != len(set(entity_columns)):
            raise ValueError("`entity_variables` contains duplicated column names.")
        if self.timestamp in entity_columns:
            raise ValueError(
                "The timestamp column cannot also be an entity column. "
                f"Got {self.timestamp!r} in both."
            )

        missing = [
            col for col in entity_columns + [self.timestamp] if col not in X.columns
        ]
        if missing:
            raise ValueError(
                "The following entity or timestamp columns are not present in "
                f"the dataframe: {missing}. Available columns: "
                f"{X.columns.tolist()}."
            )

    def _set_variables(self, X: pd.DataFrame) -> None:
        reserved = set(self._entity_columns_list()) | {self.timestamp}

        if self.variables is None:
            candidates = [
                col
                for col in X.select_dtypes(include="number").columns
                if col not in reserved
            ]
            if not candidates and not self.return_empty:
                raise TypeError(
                    "No numerical variables found in this dataframe after "
                    "excluding the entity and timestamp columns. Cast the "
                    "variables as numerical or set `return_empty=True`."
                )
            self.variables_ = candidates
        else:
            self.variables_ = check_numerical_variables(X, self.variables)
            overlap = [col for col in self.variables_ if col in reserved]
            if overlap:
                raise ValueError(
                    "Entity and timestamp columns cannot be used as feature "
                    f"variables: {overlap}."
                )

    def _entity_index(self, X: pd.DataFrame):
        entity_columns = self._entity_columns_list()
        if len(entity_columns) == 1:
            return pd.Index(X[entity_columns[0]])
        return pd.MultiIndex.from_frame(X[entity_columns])

    def _parse_timestamps(self, X: pd.DataFrame) -> pd.Series:
        series = X[self.timestamp]

        if is_datetime64_any_dtype(series):
            parsed = series
        else:
            try:
                parsed = pd.to_datetime(
                    series, format=self.date_format, errors="coerce"
                )
            except (TypeError, ValueError) as err:
                raise ValueError(
                    f"Column {self.timestamp!r} could not be parsed as datetime. "
                    "Check the values or set `date_format`."
                ) from err

        if not is_datetime64_any_dtype(parsed):
            raise ValueError(
                f"Column {self.timestamp!r} could not be parsed as a single "
                "datetime type. Check for mixed time zones or value types, or "
                "set `date_format`."
            )

        bad = np.asarray(pd.isna(parsed))
        if bad.any():
            positions = np.flatnonzero(bad)[:20].tolist()
            raise ValueError(
                f"Column {self.timestamp!r} contains missing or unparseable "
                f"timestamps in {int(bad.sum())} row(s). Original row positions "
                f"(0-based, first 20): {positions}. Check the values or set "
                "`date_format`."
            )
        # Normalise the resolution so that offsets searched against the
        # timestamps compare exactly (pandas 2+ can store datetime64[us]).
        try:
            parsed = parsed.dt.as_unit("ns")
        except (AttributeError, ValueError):
            pass
        return parsed

    def _check_entity_values(self, entities) -> None:
        if isinstance(entities, pd.MultiIndex):
            missing = np.asarray(entities.to_frame(index=False).isna().any(axis=1))
        else:
            missing = np.asarray(pd.isna(entities))
        if missing.any():
            positions = np.flatnonzero(missing)[:20].tolist()
            columns = self._entity_columns_list()
            raise ValueError(
                f"The entity column(s) {columns} contain missing values in "
                f"{int(missing.sum())} row(s). Original row positions (0-based, "
                f"first 20): {positions}."
            )

    def _raise_duplicates(self, work: pd.DataFrame) -> None:
        duplicated = work.duplicated(subset=["__fe_code__", "__fe_time__"], keep=False)
        duplicated_work = work.loc[duplicated]
        n_rows = int(duplicated.sum())
        n_keys = int(
            duplicated_work.groupby(["__fe_code__", "__fe_time__"], sort=False).ngroups
        )
        positions = duplicated_work["__fe_pos__"].head(20).tolist()
        entity_columns = self._entity_columns_list()
        keys = (
            duplicated_work[entity_columns]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        time_min = duplicated_work["__fe_time__"].min()
        time_max = duplicated_work["__fe_time__"].max()
        raise ValueError(
            f"Found {n_keys} duplicated (entity, timestamp) combination(s) "
            f"covering {n_rows} row(s). Entity keys (first 10): {keys}. Time "
            f"range: {time_min} to {time_max}. Original row positions (0-based, "
            f"first 20): {positions}. Keep the duplicates with "
            "`duplicates='first'`, `duplicates='last'` or `duplicates='keep'`, "
            "or remove them before fitting."
        )

    # ------------------------------------------------------------------
    # horizon and window helpers
    # ------------------------------------------------------------------
    def _horizon_spec(self) -> Tuple[str, Union[int, pd.Timedelta]]:
        if _is_int(self.horizon):
            return "rows", int(self.horizon)
        return "time", _to_timedelta(self.horizon, "horizon")

    def _horizon_is_positive(self) -> bool:
        """Whether the horizon removes history beyond simple causality.

        A row horizon of 0 or 1 both mean "use the previous observations"; only
        a horizon of two or more skips recent history.
        """
        kind, horizon = self._horizon_spec()
        if kind == "rows":
            return int(horizon) >= 2
        return horizon > pd.Timedelta(0)

    @staticmethod
    def _block_slices(work: pd.DataFrame) -> List[Tuple[int, int]]:
        codes = work["__fe_code__"].to_numpy()
        if len(codes) == 0:
            return []
        changes: np.ndarray = np.flatnonzero(np.diff(codes) != 0) + 1
        starts: np.ndarray = np.concatenate(([0], changes))
        ends: np.ndarray = np.concatenate((changes, [len(codes)]))
        return list(zip(starts.astype(int), ends.astype(int)))

    @staticmethod
    def _block_starts(blocks: List[Tuple[int, int]], n: int) -> np.ndarray:
        starts: np.ndarray = np.empty(n, dtype=np.int64)
        for start, end in blocks:
            starts[start:end] = start
        return starts

    def _row_ends(
        self, work: pd.DataFrame, blocks: List[Tuple[int, int]]
    ) -> np.ndarray:
        """Exclusive positional end of the history allowed for every row.

        The current row is never part of its own history, and neither is any
        row that shares its timestamp. ``horizon`` is the forecast span: with
        an integer horizon ``h`` the newest usable observation is the one
        ``h`` rows before the target (``h=0`` and ``h=1`` both mean the
        immediately preceding row). With a time horizon, it is the last
        observation at or before ``timestamp - horizon``.
        """
        times = pd.DatetimeIndex(work["__fe_time__"])
        ends: np.ndarray = np.empty(len(work), dtype=np.int64)
        kind, horizon = self._horizon_spec()

        for start, end in blocks:
            block_times = times[start:end]
            strict = block_times.searchsorted(block_times, side="left") + start
            if kind == "rows":
                positions = np.arange(start, end)
                bounds = np.minimum(positions - int(horizon) + 1, strict)
            else:
                if horizon == pd.Timedelta(0):
                    bounds = strict
                else:
                    reference = block_times - horizon
                    bounds = block_times.searchsorted(reference, side="right") + start
            ends[start:end] = np.clip(bounds, start, end)
        return ends

    def _window_starts(
        self,
        work: pd.DataFrame,
        blocks: List[Tuple[int, int]],
        ends: np.ndarray,
        window: Union[int, str, pd.Timedelta],
    ) -> np.ndarray:
        """Exclusive positional start of the lookback window for every row.

        Time windows are measured on the real clock. They cover
        ``[cutoff - window, cutoff)`` where ``cutoff`` is the target timestamp
        minus a time horizon, or the target timestamp itself for a row horizon.
        The end is capped at the allowed history, so the current row and rows
        sharing its timestamp are excluded. The lower bound is inclusive, so a
        ``"7D"`` window over daily data contains the previous seven
        observations. A gap larger than the time window still leaves it empty
        instead of pulling in a distant observation. Integer windows count
        observations inside the allowed history.
        """
        times = pd.DatetimeIndex(work["__fe_time__"])
        starts: np.ndarray = np.empty(len(work), dtype=np.int64)

        if _is_int(window):
            int_window = int(window)
            if int_window <= 0:
                raise ValueError(
                    f"`window` must be a positive integer or offset. Got {window!r}."
                )
            block_starts = self._block_starts(blocks, len(work))
            starts[:] = np.maximum(block_starts, ends - int_window)
            return starts

        delta = _to_timedelta(window, "window")
        if delta <= pd.Timedelta(0):
            raise ValueError(f"`window` must be a positive offset. Got {window!r}.")

        kind, horizon = self._horizon_spec()
        for start, end in blocks:
            block_times = times[start:end]
            for local in range(end - start):
                row = start + local
                row_end = ends[row]
                if row_end <= start:
                    starts[row] = start
                    continue
                if kind == "time" and horizon > pd.Timedelta(0):
                    cutoff = block_times[local] - horizon
                else:
                    cutoff = block_times[local]
                lower = cutoff - delta
                start_index = block_times.searchsorted(lower, side="left") + start
                starts[row] = min(row_end, max(start, start_index))
        return starts

    def _resolve_min_periods(
        self, min_periods: Optional[int], window: Optional[Union[int, str]]
    ) -> int:
        if min_periods is not None:
            if not _is_int(min_periods) or min_periods < 1:
                raise ValueError(
                    "`min_periods` must be a positive integer or None. "
                    f"Got {min_periods!r} instead."
                )
            return int(min_periods)
        if window is not None and _is_int(window):
            return int(window)
        return 1

    def _check_history_available(
        self,
        work: pd.DataFrame,
        blocks: List[Tuple[int, int]],
        ends: np.ndarray,
        window: Optional[Union[int, str, pd.Timedelta]] = None,
        min_periods: Optional[int] = None,
    ) -> None:
        """Raise when the horizon leaves no usable history for any row.

        A short history (fewer observations than the window or ``min_periods``)
        is not an error: it produces NaN values, as pandas does. The error is
        reserved for a configuration where the horizon hides *all* history for
        every row, which cannot produce a single meaningful feature.
        """
        if not self._horizon_is_positive() or len(ends) == 0:
            return
        block_starts = self._block_starts(blocks, len(work))
        if np.any(ends > block_starts):
            return

        times = pd.DatetimeIndex(work["__fe_time__"])
        detail = f"horizon={self.horizon!r}"
        if window is not None:
            detail += f", window={window!r}"
        if min_periods is not None:
            detail += f", min_periods={min_periods}"
        raise ValueError(
            f"The configuration ({detail}) leaves no usable history for any row "
            f"of any entity ({int(work['__fe_code__'].nunique())} entities, time "
            f"range {times.min()} to {times.max()}). Reduce the horizon"
            + (", the window" if window is not None else "")
            + ", or provide more history."
        )

    # ------------------------------------------------------------------
    # fit / transform plumbing
    # ------------------------------------------------------------------
    def _prepare_work(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a work frame sorted by entity and time.

        The frame keeps the original index and adds three helper columns: the
        original position, the entity code and the parsed timestamp. Entity
        codes are assigned in order of first appearance, so ordering is
        deterministic for any input order.
        """
        parsed = self._parse_timestamps(X)
        entities = self._entity_index(X)
        self._check_entity_values(entities)
        codes, _ = pd.factorize(entities, sort=False)

        work = X.copy()
        work["__fe_pos__"] = np.arange(len(X), dtype=np.int64)
        work["__fe_code__"] = codes
        work["__fe_time__"] = parsed
        work = work.sort_values(
            ["__fe_code__", "__fe_time__", "__fe_pos__"],
            kind="mergesort",
        )
        return work

    def _apply_duplicates(self, work: pd.DataFrame) -> pd.DataFrame:
        if self.duplicates == "raise":
            if work.duplicated(subset=["__fe_code__", "__fe_time__"], keep=False).any():
                self._raise_duplicates(work)
            return work
        if self.duplicates in ("first", "last"):
            return work.drop_duplicates(
                subset=["__fe_code__", "__fe_time__"], keep=self.duplicates
            ).copy()
        return work

    def _set_entity_state(self, work: pd.DataFrame) -> None:
        entity_columns = self._entity_columns_list()
        if len(entity_columns) == 1:
            self.entity_values_ = pd.Index(work[entity_columns[0]].drop_duplicates())
        else:
            self.entity_values_ = pd.MultiIndex.from_frame(
                work[entity_columns].drop_duplicates()
            )
        times = pd.DatetimeIndex(work["__fe_time__"])
        self.time_range_ = (times.min(), times.max())

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Validate the data and record the entities and time range.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset.

        y: pandas Series, default=None
            Not used. You can pass None or y.
        """
        X = check_X(X)
        self._validate_columns(X)
        self._set_variables(X)

        if self.missing_values == "raise":
            self._check_na_and_inf(X)

        self._get_feature_names_in(X)

        work = self._prepare_work(X)
        work = self._apply_duplicates(work)
        self._set_entity_state(work)

        return self

    def _unseen_entity_mask(self, X: pd.DataFrame) -> np.ndarray:
        entities = self._entity_index(X)
        return ~np.asarray(entities.isin(self.entity_values_), dtype=bool)

    def _cold_start_fill(self) -> Union[None, float]:
        if self.cold_start == "nan":
            return np.nan
        if self.cold_start == "fill":
            return float(self.cold_start_value)  # type: ignore[arg-type]
        if isinstance(self.cold_start, numbers.Number) and not isinstance(
            self.cold_start, bool
        ):
            return float(self.cold_start)  # type: ignore[arg-type]
        return None

    def _check_cold_start(self, X: pd.DataFrame) -> np.ndarray:
        unseen = self._unseen_entity_mask(X)
        if not unseen.any() or self.cold_start not in (None, "raise"):
            return unseen

        keys = (
            X.loc[unseen, self._entity_columns_list()]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        times = pd.DatetimeIndex(self._parse_timestamps(X.loc[unseen]))
        raise ValueError(
            "The dataframe to transform contains entities that were not seen "
            f"during fit ({int(unseen.sum())} row(s), entity keys (first 10): "
            f"{keys}, time range {times.min()} to {times.max()}). Set "
            "`cold_start` to 'nan', 'fill' (with `cold_start_value`) or a number "
            "to define how unseen entities are handled."
        )

    def _check_transform_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [col for col in self.feature_names_in_ if col not in X.columns]
        if missing:
            raise ValueError(
                "The dataframe to transform is missing columns seen during fit: "
                f"{missing}."
            )
        return X[self.feature_names_in_]

    def _compute_features(self, work: pd.DataFrame) -> pd.DataFrame:
        """Return features indexed by ``__fe_pos__``."""
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add grouped lag, rolling or expanding features.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: pandas dataframe, shape = [n_samples, n_features + new_features]
            The dataframe with the original columns plus the new features, in
            the original row order.
        """
        check_is_fitted(self)
        X = check_X(X)
        _check_X_matches_training_df(X, self.n_features_in_)
        X = self._check_transform_columns(X)
        self._validate_columns(X)

        if self.missing_values == "raise":
            self._check_na_and_inf(X)

        unseen = self._check_cold_start(X)

        work = self._prepare_work(X)
        work = self._apply_duplicates(work)
        features = self._compute_features(work)

        # ``work`` is sorted by entity and time; sort a copy by the original
        # position to restore input order. Rows removed as duplicates are not
        # part of the output.
        order = np.argsort(work["__fe_pos__"].to_numpy(), kind="mergesort")
        positions = work["__fe_pos__"].to_numpy()[order]
        X_out = X.iloc[positions].copy()
        names = list(features.columns)

        if names:
            features = features.reindex(work["__fe_pos__"].to_numpy()).iloc[order]
            X_out[names] = features.to_numpy()

            if unseen.any():
                fill = self._cold_start_fill()
                if fill is not None:
                    X_out.loc[unseen[positions], names] = fill

        if self.drop_original:
            X_out = X_out.drop(columns=self.variables_)

        if self.drop_na and names:
            X_out = X_out.dropna(subset=names, axis=0)

        return X_out

    def _get_new_features_name(self) -> List[str]:
        raise NotImplementedError


class GroupedLagFeatures(BaseGroupedForecastTransformer):
    """
    Lag features computed independently for each entity.

    A lag feature at a given timestamp holds the value of the same variable at
    an earlier observation of the *same* entity. Observations are ordered by the
    configured timestamp, so out-of-order input is handled deterministically and
    no value ever comes from another entity.

    Lags can be defined in rows with ``periods`` (the number of prior
    observations of the entity) or in real time with ``freq`` (for example,
    ``"1D"``). With ``freq``, the value is the latest observation at or before
    ``timestamp - freq`` within the entity, so irregular sampling and gaps are
    respected. ``horizon`` adds a forecast span on top: no observation newer
    than the horizon is used.

    Parameters
    ----------
    variables: str, int, or list of strings or integers, default=None.
        The numerical variables to lag. By default, all numerical variables
        except the entity and timestamp columns.

    entity_variables: str, int, or list, default=None.
        Column or columns identifying the entity. Required.

    timestamp: str or int, default=None.
        Column with the observation time. Required.

    return_empty: bool, default=False.
        Whether to allow an empty list of numerical variables.

    periods: int or list of ints, default=1.
        Number of prior observations of the same entity to lag. Ignored when
        ``freq`` is set.

    freq: str, list of str, or None, default=None.
        Real-time lag. When set, it overrides ``periods``. Each value is the
        latest observation at or before ``timestamp - freq`` (or
        ``timestamp - horizon - freq`` when ``horizon`` is a time offset).

    fill_value: object, default=None.
        Value used for missing lag values. When None, missing values stay NaN.

    horizon: int, str, or Timedelta, default=0.
        Forecast span. With an integer, the newest usable observation is
        ``max(horizon, 1)`` prior observations back within the entity; with an
        offset, it is the latest observation at or before
        ``timestamp - horizon``. The current row is never used.

    cold_start: None, "raise", "nan", "fill", or a number, default=None.
        Policy for entities absent from the training data. The default (None)
        raises; ``"nan"`` produces missing values; ``"fill"`` uses
        ``cold_start_value``; a number is used as the fill value.

    cold_start_value: number, default=None.
        Value for ``cold_start="fill"``.

    duplicates: str, default="raise".
        How to handle repeated ``(entity, timestamp)`` pairs: ``"raise"``
        (default), ``"first"``, ``"last"`` or ``"keep"``.

    date_format: str, default=None.
        Format used to parse the timestamp column.

    missing_values: str, default="raise".
        Whether to raise or ignore missing and infinite values in ``variables``.

    drop_original: bool, default=False.
        Whether to drop the original variables from the output.

    drop_na: bool, default=False.
        Whether to drop rows with missing values in the created features.

    Attributes
    ----------
    variables_:
        The variables that were lagged.

    entity_values_:
        The entity keys observed during fit.

    time_range_:
        First and last timestamp observed during fit.

    feature_names_in_:
        List with the names of features seen during fit.

    n_features_in_:
        The number of features in the train set used in fit.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.timeseries.forecasting import GroupedLagFeatures
    >>> X = pd.DataFrame({
    ...     "entity": ["A", "A", "A", "B", "B"],
    ...     "time": pd.to_datetime([
    ...         "2022-01-01", "2022-01-02", "2022-01-04",
    ...         "2022-01-02", "2022-01-05",
    ...     ]),
    ...     "sales": [1.0, 2.0, 4.0, 10.0, 20.0],
    ... })
    >>> tr = GroupedLagFeatures(
    ...     entity_variables="entity", timestamp="time", periods=1,
    ... )
    >>> tr.fit_transform(X)[["entity", "time", "sales_lag_1"]]
      entity       time  sales_lag_1
    0      A 2022-01-01          NaN
    1      A 2022-01-02          1.0
    2      A 2022-01-04          2.0
    3      B 2022-01-02          NaN
    4      B 2022-01-05         10.0
    """

    def __init__(
        self,
        variables: Variables = None,
        entity_variables: Variables = None,
        timestamp: Union[str, int, None] = None,
        return_empty: bool = False,
        periods: Union[int, List[int]] = 1,
        freq: Union[str, List[str], None] = None,
        fill_value: Hashable = None,
        horizon: Union[int, str, pd.Timedelta] = 0,
        cold_start: Union[None, str, numbers.Number] = None,
        cold_start_value: Optional[numbers.Number] = None,
        duplicates: str = "raise",
        date_format: Optional[str] = None,
        missing_values: str = "raise",
        drop_original: bool = False,
        drop_na: bool = False,
    ) -> None:
        if not (
            _is_int(periods)
            and int(periods) > 0  # type: ignore[arg-type]
            or isinstance(periods, list)
            and len(periods) > 0
            and all(_is_int(num) and num > 0 for num in periods)
        ):
            raise ValueError(
                "`periods` must be a positive integer or a list of positive "
                f"integers. Got {periods!r} instead."
            )
        if isinstance(periods, list) and len(periods) != len(set(periods)):
            raise ValueError(f"There are duplicated periods in the list: {periods}")

        if freq is not None:
            freqs = _as_list(freq)
            if len(freqs) == 0 or not all(isinstance(fr, str) for fr in freqs):
                raise ValueError(
                    "`freq` must be a string or a list of strings. "
                    f"Got {freq!r} instead."
                )
            if len(freqs) != len(set(freqs)):
                raise ValueError(
                    f"There are duplicated freq values in the list: {freq}"
                )
            for fr in freqs:
                _to_timedelta(fr, "freq")

        super().__init__(
            variables=variables,
            entity_variables=entity_variables,
            timestamp=timestamp,
            return_empty=return_empty,
            horizon=horizon,
            cold_start=cold_start,
            cold_start_value=cold_start_value,
            duplicates=duplicates,
            date_format=date_format,
            missing_values=missing_values,
            drop_original=drop_original,
            drop_na=drop_na,
        )
        self.periods = periods
        self.freq = freq
        self.fill_value = fill_value

    def _get_new_features_name(self) -> List[str]:
        if self.freq is not None:
            specs = _as_list(self.freq)
            return [
                f"{variable}_lag_{_offset_label(spec)}"
                for spec in specs
                for variable in self.variables_
            ]
        specs = _as_list(self.periods)
        return [
            f"{variable}_lag_{spec}" for spec in specs for variable in self.variables_
        ]

    def _specs(self) -> List[Tuple[str, Union[int, pd.Timedelta]]]:
        if self.freq is not None:
            return [("freq", _to_timedelta(fr, "freq")) for fr in _as_list(self.freq)]
        return [("rows", int(pr)) for pr in _as_list(self.periods)]

    def _source_positions(
        self,
        work: pd.DataFrame,
        blocks: List[Tuple[int, int]],
        ends: np.ndarray,
    ) -> List[np.ndarray]:
        """One source row per feature block, or -1 when there is no value.

        Row-based lags count previous observations inside the allowed history.
        Time-based lags look up the latest observation at or before the lag
        distance measured from the forecast cutoff.
        """
        times = pd.DatetimeIndex(work["__fe_time__"])
        kind, horizon = self._horizon_spec()
        sources = []

        for spec_kind, spec in self._specs():
            source: np.ndarray = np.full(len(work), -1, dtype=np.int64)
            for start, end in blocks:
                block_times = times[start:end]
                for local in range(end - start):
                    row = start + local
                    row_end = ends[row]
                    if row_end <= start:
                        continue
                    if spec_kind == "rows":
                        candidate_row = int(row_end) - int(spec)
                        if candidate_row >= start:
                            source[row] = candidate_row
                        continue

                    # A real-time lag is measured back from the target
                    # timestamp. A time-based forecast horizon is an
                    # additional upper bound on how recent the value can be;
                    # a row-based horizon is enforced by capping the source to
                    # the allowed history below.
                    reference = block_times[local] - spec
                    if kind == "time":
                        reference = min(reference, block_times[local] - horizon)
                    local_source = block_times.searchsorted(reference, side="right") - 1
                    local_source = min(local_source, row_end - start - 1)
                    if local_source >= 0:
                        source[row] = start + local_source
            sources.append(source)
        return sources

    def _compute_features(self, work: pd.DataFrame) -> pd.DataFrame:
        names = self._get_new_features_name()
        if not names:
            return pd.DataFrame(index=work["__fe_pos__"])

        blocks = self._block_slices(work)
        ends = self._row_ends(work, blocks)
        self._check_history_available(work, blocks, ends)
        sources = self._source_positions(work, blocks, ends)

        values = work[self.variables_].to_numpy(dtype=float)
        n_rows = len(work)
        columns = []
        for source in sources:
            valid = source >= 0
            block = np.full((n_rows, len(self.variables_)), np.nan)
            block[valid] = values[source[valid]]
            columns.append(block)

        data = np.concatenate(columns, axis=1)
        if self.fill_value is not None:
            data = np.where(np.isnan(data), self.fill_value, data)

        return pd.DataFrame(data, index=work["__fe_pos__"], columns=names)


class GroupedWindowFeatures(BaseGroupedForecastTransformer):
    """
    Rolling window features computed independently for each entity.

    Windows are anchored on the observations allowed by ``horizon`` and never
    cross an entity boundary. Integer windows count observations; offset
    windows (for example ``"2D"``) use real time, so gaps and irregular
    sampling are reflected in the result instead of being skipped by row
    position. When entities use different sampling frequencies, each entity is
    aligned on its own timestamps; no resampling to a common grid is performed.

    For supported aggregation functions, see the pandas
    `rolling window functions
    <https://pandas.pydata.org/docs/reference/window.html>`_.

    Parameters
    ----------
    variables: str, int, or list of strings or integers, default=None.
        The numerical variables used in the windows.

    entity_variables: str, int, or list, default=None.
        Column or columns identifying the entity. Required.

    timestamp: str or int, default=None.
        Column with the observation time. Required.

    return_empty: bool, default=False.
        Whether to allow an empty list of numerical variables.

    window: int, str, Timedelta, or list, default=3.
        Size of the rolling window. An integer counts observations; an offset
        such as ``"2D"`` or ``"45min"`` measures real time. A list creates
        features for each window.

    min_periods: int, default=None.
        Minimum number of observations required to emit a value. When None it
        defaults to the integer window size, or to 1 for offset windows, which
        matches pandas. Rows with fewer observations get NaN.

    functions: str or list of str, default="mean".
        Aggregations applied inside each window.

    horizon: int, str, or Timedelta, default=0.
        Forecast span. With an integer, the newest usable observation is
        ``max(horizon, 1)`` prior observations back within the entity; with an
        offset, it is the latest observation at or before
        ``timestamp - horizon``. The current row is never used.

    cold_start: None, "raise", "nan", "fill", or a number, default=None.
        Policy for entities absent from the training data. The default (None)
        raises; ``"nan"`` produces missing values; ``"fill"`` uses
        ``cold_start_value``; a number is used as the fill value.

    cold_start_value: number, default=None.
        Value for ``cold_start="fill"``.

    duplicates: str, default="raise".
        How to handle repeated ``(entity, timestamp)`` pairs.

    date_format: str, default=None.
        Format used to parse the timestamp column.

    missing_values: str, default="raise".
        Whether to raise or ignore missing and infinite values in ``variables``.

    drop_original: bool, default=False.
        Whether to drop the original variables from the output.

    drop_na: bool, default=False.
        Whether to drop rows with missing values in the created features.

    Attributes
    ----------
    variables_:
        The variables used in the windows.

    entity_values_:
        The entity keys observed during fit.

    time_range_:
        First and last timestamp observed during fit.

    feature_names_in_:
        List with the names of features seen during fit.

    n_features_in_:
        The number of features in the train set used in fit.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.timeseries.forecasting import GroupedWindowFeatures
    >>> X = pd.DataFrame({
    ...     "entity": ["A", "A", "A", "B", "B"],
    ...     "time": pd.to_datetime([
    ...         "2022-01-01", "2022-01-02", "2022-01-04",
    ...         "2022-01-02", "2022-01-05",
    ...     ]),
    ...     "sales": [1.0, 2.0, 4.0, 10.0, 20.0],
    ... })
    >>> tr = GroupedWindowFeatures(
    ...     entity_variables="entity", timestamp="time", window=2,
    ... )
    >>> tr.fit_transform(X)[["entity", "sales_window_2_mean"]]
      entity  sales_window_2_mean
    0      A                  NaN
    1      A                  NaN
    2      A                  1.5
    3      B                  NaN
    4      B                  NaN
    """

    def __init__(
        self,
        variables: Variables = None,
        entity_variables: Variables = None,
        timestamp: Union[str, int, None] = None,
        return_empty: bool = False,
        window: Union[int, str, pd.Timedelta, List] = 3,
        min_periods: Optional[int] = None,
        functions: Union[str, List[str]] = "mean",
        horizon: Union[int, str, pd.Timedelta] = 0,
        cold_start: Union[None, str, numbers.Number] = None,
        cold_start_value: Optional[numbers.Number] = None,
        duplicates: str = "raise",
        date_format: Optional[str] = None,
        missing_values: str = "raise",
        drop_original: bool = False,
        drop_na: bool = False,
    ) -> None:
        windows = _as_list(window)
        if len(windows) == 0:
            raise ValueError("`window` cannot be empty.")
        for win in windows:
            if _is_int(win):
                if win <= 0:
                    raise ValueError(
                        "`window` must be a positive integer or time offset. "
                        f"Got {win!r} instead."
                    )
            elif _is_offset(win):
                delta = _to_timedelta(win, "window")
                if delta <= pd.Timedelta(0):
                    raise ValueError(
                        "`window` must be a positive time offset. "
                        f"Got {win!r} instead."
                    )
            else:
                raise ValueError(
                    "`window` must be a positive integer, a time offset such as "
                    f"'2D', or a list of those. Got {win!r} instead."
                )
        if len(windows) != len(set(windows)):
            raise ValueError(f"There are duplicated windows in the list: {window}")

        if not isinstance(functions, (str, list)) or not all(
            isinstance(val, str) for val in functions
        ):
            raise ValueError(
                "`functions` must be a string or a list of strings. "
                f"Got {functions!r} instead."
            )
        if isinstance(functions, list) and len(functions) != len(set(functions)):
            raise ValueError(f"There are duplicated functions in the list: {functions}")

        if min_periods is not None and (not _is_int(min_periods) or min_periods < 1):
            raise ValueError(
                "`min_periods` must be a positive integer or None. "
                f"Got {min_periods!r} instead."
            )

        super().__init__(
            variables=variables,
            entity_variables=entity_variables,
            timestamp=timestamp,
            return_empty=return_empty,
            horizon=horizon,
            cold_start=cold_start,
            cold_start_value=cold_start_value,
            duplicates=duplicates,
            date_format=date_format,
            missing_values=missing_values,
            drop_original=drop_original,
            drop_na=drop_na,
        )
        self.window = window
        self.min_periods = min_periods
        self.functions = functions

    def _get_new_features_name(self) -> List[str]:
        functions = _as_list(self.functions)
        return [
            f"{variable}_window_{_offset_label(win)}_{function}"
            for win in _as_list(self.window)
            for variable in self.variables_
            for function in functions
        ]

    def _compute_features(self, work: pd.DataFrame) -> pd.DataFrame:
        names = self._get_new_features_name()
        if not names:
            return pd.DataFrame(index=work["__fe_pos__"])

        blocks = self._block_slices(work)
        ends = self._row_ends(work, blocks)
        values = work[self.variables_]

        frames = []
        for win in _as_list(self.window):
            starts = self._window_starts(work, blocks, ends, win)
            min_periods = self._resolve_min_periods(self.min_periods, win)
            self._check_history_available(
                work, blocks, ends, window=win, min_periods=min_periods
            )
            indexer = _GroupedWindowIndexer(starts, ends)
            raw = values.rolling(indexer, min_periods=min_periods).agg(self.functions)
            raw.columns = self._rolling_names(win)
            frames.append(raw)

        result = pd.concat(frames, axis=1)
        result.index = work["__fe_pos__"]
        return result[names]

    def _rolling_names(self, window) -> List[str]:
        functions = _as_list(self.functions)
        return [
            f"{variable}_window_{_offset_label(window)}_{function}"
            for variable in self.variables_
            for function in functions
        ]


class GroupedExpandingWindowFeatures(BaseGroupedForecastTransformer):
    """
    Expanding window features computed independently for each entity.

    For each row, the expanding window contains all observations of the same
    entity that are allowed by ``horizon``: never the current row, never a row
    sharing its timestamp, and never data from another entity. The window
    starts at the entity's first observation and grows with each new row.

    For supported aggregation functions, see the pandas
    `expanding window functions
    <https://pandas.pydata.org/docs/reference/window.html>`_.

    Parameters
    ----------
    variables: str, int, or list of strings or integers, default=None.
        The numerical variables used in the windows.

    entity_variables: str, int, or list, default=None.
        Column or columns identifying the entity. Required.

    timestamp: str or int, default=None.
        Column with the observation time. Required.

    return_empty: bool, default=False.
        Whether to allow an empty list of numerical variables.

    min_periods: int, default=None.
        Minimum number of observations required to emit a value. When None it
        defaults to 1, matching pandas. Rows with fewer observations get NaN.

    functions: str or list of str, default="mean".
        Aggregations applied inside each window.

    horizon: int, str, or Timedelta, default=0.
        Forecast span. With an integer, the newest usable observation is
        ``max(horizon, 1)`` prior observations back within the entity; with an
        offset, it is the latest observation at or before
        ``timestamp - horizon``. The current row is never used.

    cold_start: None, "raise", "nan", "fill", or a number, default=None.
        Policy for entities absent from the training data. The default (None)
        raises; ``"nan"`` produces missing values; ``"fill"`` uses
        ``cold_start_value``; a number is used as the fill value.

    cold_start_value: number, default=None.
        Value for ``cold_start="fill"``.

    duplicates: str, default="raise".
        How to handle repeated ``(entity, timestamp)`` pairs.

    date_format: str, default=None.
        Format used to parse the timestamp column.

    missing_values: str, default="raise".
        Whether to raise or ignore missing and infinite values in ``variables``.

    drop_original: bool, default=False.
        Whether to drop the original variables from the output.

    drop_na: bool, default=False.
        Whether to drop rows with missing values in the created features.

    Attributes
    ----------
    variables_:
        The variables used in the windows.

    entity_values_:
        The entity keys observed during fit.

    time_range_:
        First and last timestamp observed during fit.

    feature_names_in_:
        List with the names of features seen during fit.

    n_features_in_:
        The number of features in the train set used in fit.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.timeseries.forecasting import (
    ...     GroupedExpandingWindowFeatures,
    ... )
    >>> X = pd.DataFrame({
    ...     "entity": ["A", "A", "A", "B", "B"],
    ...     "time": pd.to_datetime([
    ...         "2022-01-01", "2022-01-02", "2022-01-04",
    ...         "2022-01-02", "2022-01-05",
    ...     ]),
    ...     "sales": [1.0, 2.0, 4.0, 10.0, 20.0],
    ... })
    >>> tr = GroupedExpandingWindowFeatures(
    ...     entity_variables="entity", timestamp="time",
    ... )
    >>> tr.fit_transform(X)[["entity", "sales_expanding_mean"]]
      entity  sales_expanding_mean
    0      A                   NaN
    1      A                   1.0
    2      A                   1.5
    3      B                   NaN
    4      B                  10.0
    """

    def __init__(
        self,
        variables: Variables = None,
        entity_variables: Variables = None,
        timestamp: Union[str, int, None] = None,
        return_empty: bool = False,
        min_periods: Optional[int] = None,
        functions: Union[str, List[str]] = "mean",
        horizon: Union[int, str, pd.Timedelta] = 0,
        cold_start: Union[None, str, numbers.Number] = None,
        cold_start_value: Optional[numbers.Number] = None,
        duplicates: str = "raise",
        date_format: Optional[str] = None,
        missing_values: str = "raise",
        drop_original: bool = False,
        drop_na: bool = False,
    ) -> None:
        if not isinstance(functions, (str, list)) or not all(
            isinstance(val, str) for val in functions
        ):
            raise ValueError(
                "`functions` must be a string or a list of strings. "
                f"Got {functions!r} instead."
            )
        if isinstance(functions, list) and len(functions) != len(set(functions)):
            raise ValueError(f"There are duplicated functions in the list: {functions}")

        if min_periods is not None and (not _is_int(min_periods) or min_periods < 1):
            raise ValueError(
                "`min_periods` must be a positive integer or None. "
                f"Got {min_periods!r} instead."
            )

        super().__init__(
            variables=variables,
            entity_variables=entity_variables,
            timestamp=timestamp,
            return_empty=return_empty,
            horizon=horizon,
            cold_start=cold_start,
            cold_start_value=cold_start_value,
            duplicates=duplicates,
            date_format=date_format,
            missing_values=missing_values,
            drop_original=drop_original,
            drop_na=drop_na,
        )
        self.min_periods = min_periods
        self.functions = functions

    def _get_new_features_name(self) -> List[str]:
        return [
            f"{variable}_expanding_{function}"
            for variable in self.variables_
            for function in _as_list(self.functions)
        ]

    def _compute_features(self, work: pd.DataFrame) -> pd.DataFrame:
        names = self._get_new_features_name()
        if not names:
            return pd.DataFrame(index=work["__fe_pos__"])

        blocks = self._block_slices(work)
        ends = self._row_ends(work, blocks)
        starts = self._block_starts(blocks, len(work))
        min_periods = self._resolve_min_periods(self.min_periods, None)
        self._check_history_available(work, blocks, ends, min_periods=min_periods)

        indexer = _GroupedWindowIndexer(starts, ends)
        raw = (
            work[self.variables_]
            .rolling(indexer, min_periods=min_periods)
            .agg(self.functions)
        )
        raw.columns = [
            f"{variable}_expanding_{function}"
            for variable in self.variables_
            for function in _as_list(self.functions)
        ]
        raw.index = work["__fe_pos__"]
        return raw[names]


__all__ = [
    "GroupedLagFeatures",
    "GroupedWindowFeatures",
    "GroupedExpandingWindowFeatures",
]
