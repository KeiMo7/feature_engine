# Authors: KeiMo7
# License: BSD 3 clause

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_consistent_length, check_is_fitted

from feature_engine._base_transformers.base_numerical import BaseNumericalTransformer
from feature_engine._check_init_parameters.check_init_input_params import (
    _check_return_empty_is_bool,
)
from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine._docstrings.fit_attributes import (
    _binner_dict_docstring,
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters.all_transformers import (
    _return_empty_docstring,
    _variables_numerical_docstring,
)
from feature_engine._docstrings.methods import _fit_transform_docstring
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import (
    _check_X_matches_training_df,
    check_X,
    check_y,
)
from feature_engine.tags import _return_tags
from feature_engine.variable_handling import (
    check_numerical_variables,
    find_numerical_variables,
)

_MONOTONIC_OPTIONS = ("auto", "increasing", "decreasing")
_BIN_OUTPUT_OPTIONS = ("bin_number", "woe", "boundaries")

# smoothing constant added to bad/good counts of non-empty bins when at least
# one non-empty bin has zero bads or zero goods, to avoid infinite WOE values.
_SMOOTHING = 0.5


def _check_contains_inf_by_column(
    X: pd.DataFrame, variables: List[Union[str, int]]
) -> None:
    """Reject infinite values, naming the offending columns."""
    inf_vars = [var for var in variables if np.isinf(X[var]).any()]
    if inf_vars:
        raise ValueError(
            "The following variables contain inf values, which cannot be "
            f"sorted into bins: {inf_vars}. Replace or remove those values "
            "before using this transformer."
        )


def _prebin(
    x: np.ndarray, y: np.ndarray, initial_bins: int
) -> List[List[float]]:
    """Sort non-missing values into fine equal-frequency pre-bins.

    Returns a list of blocks ``[count, bad, upper]`` where ``upper`` is the
    maximum value contained in the block (a value observed in the data, so
    final boundaries are always real, reproducible cut points).
    """
    quantiles = np.linspace(0.0, 1.0, initial_bins + 1)
    edges = np.unique(np.quantile(x, quantiles))
    if edges.size == 1:
        # constant variable: a single block
        return [[float(x.size), float(y.sum()), float(edges[0])]]

    inner = edges[1:-1]
    # right=True matches pd.cut's default: values equal to an edge fall into
    # the lower bin, so fit-time statistics and transform() agree.
    idx = np.digitize(x, inner, right=True)
    n_bins = edges.size - 1
    counts = np.bincount(idx, minlength=n_bins)
    bads = np.bincount(idx, weights=y, minlength=n_bins)
    blocks = [
        [float(counts[i]), float(bads[i]), float(edges[i + 1])]
        for i in range(n_bins)
        if counts[i] > 0
    ]
    return blocks


def _rate(block: List[float]) -> float:
    return block[1] / block[0]


def _merge(blocks: List[List[float]], i: int, j: int) -> None:
    """Merge adjacent blocks i and j (j = i +/- 1) in place."""
    lo, hi = min(i, j), max(i, j)
    merged = [
        blocks[lo][0] + blocks[hi][0],
        blocks[lo][1] + blocks[hi][1],
        blocks[hi][2],
    ]
    blocks[lo] = merged
    del blocks[hi]


def _pava(blocks: List[List[float]], increasing: bool) -> List[List[float]]:
    """Pool Adjacent Violators Algorithm.

    Merges adjacent pre-bins until the bad rate is monotonically
    non-decreasing (increasing=True) or non-increasing (increasing=False)
    across the resulting blocks. Deterministic.
    """
    stack: List[List[float]] = []
    for block in blocks:
        stack.append(list(block))
        while len(stack) >= 2:
            r_prev, r_last = _rate(stack[-2]), _rate(stack[-1])
            violates = r_prev > r_last if increasing else r_prev < r_last
            if not violates:
                break
            _merge(stack, len(stack) - 2, len(stack) - 1)
    return stack


def _enforce_constraints(
    blocks: List[List[float]], min_count: float, max_bins: int
) -> List[List[float]]:
    """Merge blocks until every block has >= min_count samples and there are
    at most max_bins blocks.

    Merging adjacent blocks never breaks monotonicity (the pooled rate lies
    between the merged rates), so a monotone sequence stays monotone.
    Deterministic: ties are always resolved towards the leftmost block.
    """
    blocks = [list(b) for b in blocks]
    while len(blocks) > 1:
        small = [i for i, b in enumerate(blocks) if b[0] < min_count]
        if small:
            i = small[0]
            if i == 0:
                j = 1
            elif i == len(blocks) - 1:
                j = i - 1
            else:
                d_left = abs(_rate(blocks[i]) - _rate(blocks[i - 1]))
                d_right = abs(_rate(blocks[i]) - _rate(blocks[i + 1]))
                j = i - 1 if d_left <= d_right else i + 1
            _merge(blocks, i, j)
            continue
        if len(blocks) > max_bins:
            diffs = [
                abs(_rate(blocks[i]) - _rate(blocks[i + 1]))
                for i in range(len(blocks) - 1)
            ]
            i = int(np.argmin(diffs))
            _merge(blocks, i, i + 1)
            continue
        break
    return blocks


def _compute_woe_iv(
    counts: np.ndarray, bads: np.ndarray
) -> "tuple[np.ndarray, np.ndarray]":
    """Compute weight of evidence and information value per bin.

    Bins with zero observations (e.g. an absent missing bin) get woe 0.0 and
    iv 0.0. If any non-empty bin has zero bads or zero goods, all non-empty
    bins are smoothed with +0.5 to avoid infinite values.
    """
    counts = np.asarray(counts, dtype=float)
    bads = np.asarray(bads, dtype=float)
    goods = counts - bads
    non_empty = counts > 0

    woe = np.zeros(counts.size)
    iv = np.zeros(counts.size)
    if not non_empty.any():
        return woe, iv

    b = bads[non_empty]
    g = goods[non_empty]
    if (b == 0).any() or (g == 0).any():
        b = b + _SMOOTHING
        g = g + _SMOOTHING

    bad_dist = b / b.sum()
    good_dist = g / g.sum()
    woe[non_empty] = np.log(bad_dist / good_dist)
    iv[non_empty] = (bad_dist - good_dist) * woe[non_empty]
    return woe, iv


@Substitution(
    variables=_variables_numerical_docstring,
    return_empty=_return_empty_docstring,
    binner_dict_=_binner_dict_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
)
class MonotonicDiscretiser(BaseNumericalTransformer):
    """
    The MonotonicDiscretiser() sorts numerical variables into intervals whose
    bad rate (the mean of a binary target) is monotonic, as required by
    credit-risk scorecards. It is designed for variables where the event rate
    should increase or decrease steadily across the bins.

    The discretiser first sorts the non-missing values into fine
    equal-frequency pre-bins, then merges adjacent pre-bins with the Pool
    Adjacent Violators Algorithm (PAVA) until the bad rate is monotonic, and
    finally merges further bins until every bin holds at least `min_bin_size`
    observations and there are at most `max_bins` bins. Merging adjacent bins
    never breaks monotonicity, so the final bins are always the closest
    feasible solution to the requested constraints.

    Missing values are kept in a separate bin (labelled -1 or "Missing") that
    is never merged with the interval bins but is included in the weight of
    evidence (WOE) and information value (IV) statistics.

    The bin limits, the WOE values and the missing-value strategy are all
    learned in one single `fit()` call and frozen afterwards: `transform()`
    only uses the stored boundaries and never looks at the target, so the
    same value always lands in the same bin after deployment. Values outside
    the training range fall into the first or last bin (the outer boundaries
    are -inf / inf); those boundary bins are flagged in the binning table.

    The MonotonicDiscretiser() works only with numerical variables. You can
    pass a list with the variables to transform, or it will automatically
    select all numerical variables.

    Parameters
    ----------
    {variables}

    {return_empty}

    max_bins: int, default=5
        Maximum number of interval bins per variable, not counting the
        missing-value bin.

    min_bin_size: int or float, default=0.05
        Minimum number of non-missing observations per interval bin. If int,
        it is an absolute count; if float in (0, 1), it is a fraction of the
        non-missing observations of the variable. If the variable does not
        have enough observations to satisfy the minimum, fit() raises an
        error stating the shortfall and a suggested value.

    monotonic: str or None, default="auto"
        Direction of the monotonic constraint on the bad rate: "increasing",
        "decreasing", or "auto", which fits both directions and keeps the one
        with the highest information value (ties resolve to "increasing").
        If None, no monotonic constraint is applied and the discretiser
        degrades to a reproducible equal-frequency binning that still honours
        `min_bin_size` and `max_bins`.

    initial_bins: int, default=20
        Number of equal-frequency pre-bins used before merging. Must be >= 2.

    bin_output: str, default="bin_number"
        What transform() returns: "bin_number" (interval bins are numbered
        0..k-1 in ascending order, the missing bin is -1), "woe" (the weight
        of evidence learned in fit()) or "boundaries" (the interval limits as
        strings, the missing bin is "Missing").

    missing_bin: bool, default=True
        Whether missing values are placed in their own bin. If False, missing
        values are returned as NaN by transform().

    precision: int, default=3
        Number of decimals used to display the interval limits when
        `bin_output="boundaries"`.

    Attributes
    ----------
    {binner_dict_}

    binning_table_:
        Dictionary with one pandas dataframe per variable. Each dataframe has
        one row per interval bin plus a "Missing" row, with the columns
        `bin`, `lower`, `upper`, `count`, `bad`, `good`, `bad_rate`, `woe`,
        `iv` and `is_boundary` (True for the first and last interval bins,
        which also catch out-of-range values).

    iv_:
        Dictionary with the total information value per variable.

    monotonic_direction_:
        Dictionary with the monotonic direction effectively used per
        variable ("increasing" or "decreasing"; None if `monotonic=None`).

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Learn the monotonic bin limits, WOE values and missing-value
        strategy. Requires the binary target y.

    {fit_transform}

    transform:
        Sort the variables into the bins learned during fit. Does not use y.

    Notes
    -----
    Fitting is fully deterministic: equal inputs and equal parameters always
    produce identical boundaries and bin order. Constant variables get a
    single bin covering (-inf, inf). Variables with too few non-missing
    observations to satisfy `min_bin_size` are rejected with an informative
    error. Infinite values are rejected before fitting, naming the column.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from feature_engine.discretisation import MonotonicDiscretiser
    >>> rng = np.random.default_rng(42)
    >>> x = rng.normal(0, 1, 500)
    >>> y = pd.Series(rng.binomial(1, 1 / (1 + np.exp(-x))))
    >>> X = pd.DataFrame({"score": x})
    >>> md = MonotonicDiscretiser(max_bins=4, monotonic="increasing")
    >>> md.fit(X, y)
    MonotonicDiscretiser(max_bins=4, monotonic='increasing')
    >>> table = md.binning_table_["score"]
    >>> table["bad_rate"].dropna().is_monotonic_increasing
    True
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_empty: bool = False,
        max_bins: int = 5,
        min_bin_size: Union[int, float] = 0.05,
        monotonic: Optional[str] = "auto",
        initial_bins: int = 20,
        bin_output: str = "bin_number",
        missing_bin: bool = True,
        precision: int = 3,
    ) -> None:

        if not isinstance(max_bins, int) or isinstance(max_bins, bool) or max_bins < 1:
            raise ValueError(
                f"max_bins must be a positive integer. Got {max_bins} instead."
            )

        if isinstance(min_bin_size, bool) or not isinstance(
            min_bin_size, (int, float)
        ):
            raise ValueError(
                "min_bin_size must be a positive integer or a float in (0, 1). "
                f"Got {min_bin_size} instead."
            )
        if isinstance(min_bin_size, float) and not 0 < min_bin_size < 1:
            raise ValueError(
                "If float, min_bin_size must be in the open interval (0, 1). "
                f"Got {min_bin_size} instead."
            )
        if isinstance(min_bin_size, int) and min_bin_size < 1:
            raise ValueError(
                "If int, min_bin_size must be >= 1. "
                f"Got {min_bin_size} instead."
            )

        if monotonic is not None and monotonic not in _MONOTONIC_OPTIONS:
            raise ValueError(
                f"monotonic takes values {_MONOTONIC_OPTIONS} or None. "
                f"Got {monotonic} instead."
            )

        if (
            not isinstance(initial_bins, int)
            or isinstance(initial_bins, bool)
            or initial_bins < 2
        ):
            raise ValueError(
                f"initial_bins must be an integer >= 2. Got {initial_bins} instead."
            )

        if bin_output not in _BIN_OUTPUT_OPTIONS:
            raise ValueError(
                f"bin_output takes values {_BIN_OUTPUT_OPTIONS}. "
                f"Got {bin_output} instead."
            )

        if not isinstance(missing_bin, bool):
            raise ValueError(
                f"missing_bin must be True or False. Got {missing_bin} instead."
            )

        if not isinstance(precision, int) or precision < 1:
            raise ValueError(
                f"precision must be a positive integer. Got {precision} instead."
            )

        _check_return_empty_is_bool(return_empty)

        self.variables = _check_variables_input_value(variables)
        self.return_empty = return_empty
        self.max_bins = max_bins
        self.min_bin_size = min_bin_size
        self.monotonic = monotonic
        self.initial_bins = initial_bins
        self.bin_output = bin_output
        self.missing_bin = missing_bin
        self.precision = precision

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Learn the monotonic bin limits, the weight of evidence per bin and
        the missing-value strategy. Everything is frozen after this call.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset. Can be the entire dataframe, not just the
            variables to be transformed.

        y: pandas series
            The binary target (0/1), where 1 is the event ("bad"). Required.
        """
        if y is None:
            raise ValueError(
                "fit requires the binary target y to compute the bad rate per "
                "bin, but got y=None."
            )

        # check input dataframe and select numerical variables. Unlike other
        # discretisers, missing values are allowed: they form their own bin.
        X = check_X(X)
        if self.variables is None:
            variables_ = find_numerical_variables(X, return_empty=self.return_empty)
        else:
            variables_ = check_numerical_variables(X, self.variables)

        # reject inf values before fitting, locating the offending columns
        _check_contains_inf_by_column(X, variables_)

        # check the target: binary 0/1 with both classes present
        y = check_y(y, y_numeric=True)
        check_consistent_length(X, y)
        check_classification_targets(y)
        y_values = np.asarray(y).ravel()
        unique_y = set(np.unique(y_values))
        if not unique_y.issubset({0, 1}):
            raise ValueError(
                "The MonotonicDiscretiser requires a binary target with values "
                f"0 and 1, where 1 is the event (bad). Got {sorted(unique_y)} "
                "instead."
            )
        if len(unique_y) < 2:
            raise ValueError(
                "The MonotonicDiscretiser requires both classes (0 and 1) to be "
                "present in y, otherwise the bad rate and WOE are undefined."
            )
        y_arr = y_values.astype(float)

        binner_dict_: Dict = {}
        binning_table_: Dict = {}
        iv_: Dict = {}
        monotonic_direction_: Dict = {}

        for var in variables_:
            x_full = X[var].to_numpy(dtype=float)
            not_missing = ~np.isnan(x_full)
            x = x_full[not_missing]
            yv = y_arr[not_missing]
            n = x.size

            min_count = self._min_count(n)

            if n < max(min_count, 1):
                if isinstance(self.min_bin_size, int) and n >= 1:
                    suggestion = f"min_bin_size={n}"
                elif isinstance(self.min_bin_size, float):
                    suggestion = f"min_bin_size={self.min_bin_size / 2}"
                else:
                    suggestion = "provide more data"
                raise ValueError(
                    f"Variable '{var}' has {n} non-missing observations, which "
                    f"is not enough to satisfy the minimum bin size of "
                    f"{min_count} samples per bin (shortfall: "
                    f"{min_count - n}). Suggestion: reduce min_bin_size (e.g. "
                    f"{suggestion}) or provide more data. No bins were "
                    "created for this variable."
                )

            # fine equal-frequency pre-bins
            blocks = _prebin(x, yv, self.initial_bins)

            # monotonic merge
            if self.monotonic is None:
                direction = None
            elif self.monotonic == "auto":
                direction = self._select_direction(blocks, min_count)
            else:
                direction = self.monotonic

            if direction is not None:
                blocks = _pava(blocks, increasing=(direction == "increasing"))

            # enforce min bin size and max number of bins
            blocks = _enforce_constraints(blocks, min_count, self.max_bins)

            # final boundaries: real cut points, outer edges open to catch
            # out-of-range values in the boundary bins
            edges = [float("-inf")] + [b[2] for b in blocks[:-1]] + [float("inf")]

            # statistics per interval bin (same binning rule as transform)
            idx = np.digitize(x, edges[1:-1], right=True)
            k = len(blocks)
            counts: np.ndarray = np.bincount(idx, minlength=k).astype(float)
            bads: np.ndarray = np.bincount(idx, weights=yv, minlength=k).astype(float)

            # the missing bin participates in the statistics
            n_missing = int((~not_missing).sum())
            counts_all: np.ndarray = np.append(counts, float(n_missing))
            bads_all: np.ndarray = np.append(bads, float(y_arr[~not_missing].sum()))

            woe, iv = _compute_woe_iv(counts_all, bads_all)

            binner_dict_[var] = edges
            monotonic_direction_[var] = direction
            iv_[var] = float(iv.sum())
            binning_table_[var] = self._build_table(
                edges, counts_all, bads_all, woe, iv, n_missing
            )

        self.binner_dict_ = binner_dict_
        self.binning_table_ = binning_table_
        self.iv_ = iv_
        self.monotonic_direction_ = monotonic_direction_
        self.variables_ = variables_
        self._get_feature_names_in(X)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Sort the variable values into the bins learned during fit().

        Uses only the frozen boundaries and WOE values; it never reads the
        target and never re-computes the bins, so the same value always lands
        in the same bin. Values outside the training range fall into the
        first or last (boundary) bin.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: pandas dataframe of shape = [n_samples, n_features]
            The transformed data. Depending on `bin_output`, the variables
            contain the bin number (missing = -1), the WOE of the bin, or the
            interval limits as strings (missing = "Missing").
        """
        check_is_fitted(self)

        X = check_X(X)
        _check_X_matches_training_df(X, self.n_features_in_)
        _check_contains_inf_by_column(X, self.variables_)
        X = X[self.feature_names_in_]

        for var in self.variables_:
            edges = self.binner_dict_[var]

            if self.bin_output == "boundaries":
                intervals = pd.cut(
                    X[var],
                    edges,
                    precision=self.precision,
                    include_lowest=True,
                )
                X[var] = intervals.astype(str)
                if self.missing_bin:
                    X[var] = X[var].mask(X[var].isna() | (X[var] == "nan"), "Missing")

            else:
                binned = pd.cut(X[var], edges, labels=False, include_lowest=True)
                if self.bin_output == "woe":
                    table = self.binning_table_[var]
                    woe_map = dict(
                        zip(
                            table.loc[table["bin"] != "Missing", "bin"].astype(int),
                            table.loc[table["bin"] != "Missing", "woe"],
                        )
                    )
                    X[var] = binned.map(woe_map)
                    if self.missing_bin:
                        missing_woe = float(
                            table.loc[table["bin"] == "Missing", "woe"].iloc[0]
                        )
                        X[var] = X[var].fillna(missing_woe)
                else:
                    if self.missing_bin:
                        X[var] = binned.fillna(-1).astype(np.int64)
                    else:
                        X[var] = binned

        return X

    def _min_count(self, n: int) -> int:
        """Minimum observations per bin, as an absolute count."""
        if isinstance(self.min_bin_size, int):
            return self.min_bin_size
        return int(np.ceil(self.min_bin_size * n))

    def _select_direction(
        self, blocks: List[List[float]], min_count: int
    ) -> str:
        """Pick the monotonic direction with the highest IV (auto mode).

        Ties resolve to "increasing", so the result is deterministic.
        """
        best_direction = "increasing"
        best_iv = None
        for direction in ("increasing", "decreasing"):
            candidate = _pava(blocks, increasing=(direction == "increasing"))
            candidate = _enforce_constraints(candidate, min_count, self.max_bins)
            counts = np.array([b[0] for b in candidate])
            bads = np.array([b[1] for b in candidate])
            _, iv = _compute_woe_iv(counts, bads)
            total_iv = float(iv.sum())
            if best_iv is None or total_iv > best_iv:
                best_iv = total_iv
                best_direction = direction
        return best_direction

    @staticmethod
    def _build_table(
        edges: List[float],
        counts: np.ndarray,
        bads: np.ndarray,
        woe: np.ndarray,
        iv: np.ndarray,
        n_missing: int,
    ) -> pd.DataFrame:
        """Assemble the binning table: one row per bin plus the missing bin."""
        k = len(edges) - 1
        rows = []
        for i in range(k):
            rows.append(
                {
                    "bin": i,
                    "lower": edges[i],
                    "upper": edges[i + 1],
                    "count": int(counts[i]),
                    "bad": int(bads[i]),
                    "good": int(counts[i] - bads[i]),
                    "bad_rate": bads[i] / counts[i] if counts[i] > 0 else np.nan,
                    "woe": float(woe[i]),
                    "iv": float(iv[i]),
                    "is_boundary": i == 0 or i == k - 1,
                }
            )
        rows.append(
            {
                "bin": "Missing",
                "lower": np.nan,
                "upper": np.nan,
                "count": int(n_missing),
                "bad": int(bads[-1]),
                "good": int(counts[-1] - bads[-1]),
                "bad_rate": bads[-1] / counts[-1] if counts[-1] > 0 else np.nan,
                "woe": float(woe[-1]),
                "iv": float(iv[-1]),
                "is_boundary": False,
            }
        )
        return pd.DataFrame(rows)

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "numerical"
        tags_dict["requires_y"] = True
        # WOE / IV and the bad rate are only defined for a binary target;
        # these checks feed a multiclass or otherwise non-binary target.
        binary_target_checks = {
            "check_fit_score_takes_y": "Check feeds a multiclass target; "
            "the MonotonicDiscretiser requires a binary target.",
            "check_estimators_overwrite_params": "Check feeds a multiclass "
            "target; the MonotonicDiscretiser requires a binary target.",
            "check_dont_overwrite_parameters": "Check feeds a multiclass "
            "target; the MonotonicDiscretiser requires a binary target.",
            "check_estimators_fit_returns_self": "Check feeds a multiclass "
            "target; the MonotonicDiscretiser requires a binary target.",
            "check_readonly_memmap_input": "Check feeds a multiclass target; "
            "the MonotonicDiscretiser requires a binary target.",
            "check_positive_only_tag_during_fit": "Check feeds a multiclass "
            "target; the MonotonicDiscretiser requires a binary target.",
            "check_estimators_dtypes": "Check feeds a target with labels "
            "{1, 2}; the MonotonicDiscretiser requires a binary 0/1 target.",
            "check_f_contiguous_array_estimator": "Check feeds a multiclass "
            "target; the MonotonicDiscretiser requires a binary target.",
            "check_methods_subset_invariance": "Check feeds a multiclass "
            "target; the MonotonicDiscretiser requires a binary target.",
            "check_dict_unchanged": "Check feeds a multiclass target; the "
            "MonotonicDiscretiser requires a binary target.",
            "check_fit2d_1sample": "Feature-engine raises its own informative "
            "error for a single sample instead of sklearn's message pattern.",
            "check_fit2d_1feature": "Feature-engine raises its own informative "
            "error for a single feature instead of sklearn's message pattern.",
        }
        tags_dict["_xfail_checks"].update(binary_target_checks)
        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        # missing values are a first-class bin, not an error
        tags.input_tags.allow_nan = True
        return tags
