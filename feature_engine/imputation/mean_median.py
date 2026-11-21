# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import warnings
from typing import List, Optional, Union

import pandas as pd

from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine._check_init_parameters.check_init_input_params import (
    _check_return_empty_is_bool,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _imputer_dict_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters.all_transformers import (
    _variables_numerical_docstring,
    _return_empty_docstring,
)
from feature_engine._docstrings.methods import (
    _fit_transform_docstring,
    _transform_imputers_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import (
    _check_X_matches_training_df,
    check_X,
)
from feature_engine.imputation.base_imputer import BaseImputer
from feature_engine.validation import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESERVOIR_SIZE,
    ChunkedStatisticsAccumulator,
    StatisticsValidationMixin,
)
from feature_engine.variable_handling import (
    check_numerical_variables,
    find_numerical_variables,
)


@Substitution(
    variables=_variables_numerical_docstring,
    return_empty=_return_empty_docstring,
    imputer_dict_=_imputer_dict_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    transform=_transform_imputers_docstring,
    fit_transform=_fit_transform_docstring,
)
class MeanImputer(StatisticsValidationMixin, BaseImputer):
    """
    The MeanImputer() replaces missing data by the mean or median value of the
    variable. It works only with numerical variables.

    You can pass a list of variables to impute. Alternatively, the
    MeanImputer() will automatically select all variables of type numeric in the
    training set.

    More details in the :ref:`User Guide <mean_imputer>`.

    Parameters
    ----------
    imputation_method: str, default='median'
        Desired method of imputation. Can take 'mean' or 'median'.

    {variables}

    {return_empty}

    validate_stats: bool, default=False
        Whether to enable the statistics validation. When True, fit() records
        the learned statistics together with their provenance (data scope,
        exactness, error bounds) in the attribute `statistics_snapshot_`,
        freezes them (further fit() calls raise, use refit() instead), and
        enables check_statistics() and check_drift(). When False (default),
        the transformer behaves exactly as before.

    stats_scope: str, default='full'
        Provenance label recorded for the data range used at fit time:
        'full' when fitting on the complete training set, 'sampled' when
        fitting on a sample drawn from it. The scope 'chunked' is recorded
        automatically when the statistics are learned through partial_fit().
        Only recorded; it does not change how the statistics are computed.

    random_state: int or None, default=None
        Seed for the reservoir sampling used by partial_fit() to estimate
        approximate quantiles (median). Fix it to make chunked fits
        reproducible. Has no effect on fit() or on exact statistics.

    Attributes
    ----------
    {imputer_dict_}

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    statistics_snapshot_:
        Record of the learned statistics and their provenance. Only present
        when fitted with `validate_stats=True` or through partial_fit().

    Methods
    -------
    fit:
        Learn the mean or median values.

    {fit_transform}

    {transform}

    partial_fit:
        Learn the statistics from data fed in chunks.

    refit:
        Explicitly re-learn the statistics (allowed when frozen).

    check_statistics:
        Compare the frozen statistics against another transformer or data.

    check_drift:
        Compare the distribution of new data against the fit-time one.

    save_statistics:
        Save the statistics snapshot to a JSON file.

    Examples
    --------

    >>> import pandas as pd
    >>> import numpy as np
    >>> from feature_engine.imputation import MeanImputer
    >>> X = pd.DataFrame(dict(
    >>>        x1 = [np.nan,1,1,0,np.nan],
    >>>        x2 = ["a", np.nan, "b", np.nan, "a"],
    >>>        ))
    >>> mmi = MeanImputer(imputation_method='median')
    >>> mmi.fit(X)
    >>> mmi.transform(X)
        x1   x2
    0  1.0    a
    1  1.0  NaN
    2  1.0    b
    3  0.0  NaN
    4  1.0    a
    """

    def __init__(
        self,
        imputation_method: str = "median",
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_empty: bool = False,
        validate_stats: bool = False,
        stats_scope: str = "full",
        random_state: Optional[int] = None,
    ) -> None:

        if imputation_method not in ["median", "mean"]:
            raise ValueError("imputation_method takes only values 'median' or 'mean'")

        if not isinstance(validate_stats, bool):
            raise ValueError(
                "validate_stats takes only booleans True and False. "
                f"Got {validate_stats} instead."
            )

        if stats_scope not in ("full", "sampled"):
            raise ValueError(
                "stats_scope takes only values 'full' or 'sampled'. "
                f"Got {stats_scope} instead."
            )

        self.imputation_method = imputation_method
        self.variables = _check_variables_input_value(variables)

        _check_return_empty_is_bool(return_empty)
        self.return_empty = return_empty

        self.validate_stats = validate_stats
        self.stats_scope = stats_scope
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn the mean or median values.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset.

        y: pandas series or None, default=None
            y is not needed in this imputation. You can pass None or y.
        """

        # block accidental re-fits when the statistics are frozen
        self._guard_against_refit()

        # check input dataframe
        X = check_X(X)

        # find or check for numerical variables
        if self.variables is None:
            variables_ = find_numerical_variables(X, self.return_empty)
        else:
            variables_ = check_numerical_variables(X, self.variables)

        # find imputation parameters: mean or median
        if self.imputation_method == "mean":
            imputer_dict_ = X[variables_].mean().to_dict()

        elif self.imputation_method == "median":
            imputer_dict_ = X[variables_].median().to_dict()

        self.variables_ = variables_
        self.imputer_dict_ = imputer_dict_
        self._get_feature_names_in(X)

        # record the statistics and their provenance (no-op unless
        # validate_stats=True)
        self._record_fit_statistics(
            X,
            statistics=self._get_fitted_statistics(),
            scope=self.stats_scope,
            exact={self.imputation_method: True},
            random_state=self.random_state,
        )

        return self

    def partial_fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        reservoir_size: int = DEFAULT_RESERVOIR_SIZE,
    ):
        """
        Learn the imputation values from data fed in chunks.

        Use this method when the training data does not fit in memory and is
        fed in several chunks. Call partial_fit() once per chunk; the
        statistics are updated after each chunk and transform() can be used
        at any point.

        The mean is **exact**: it is computed from running sums, so it equals
        the mean of the full data (up to floating-point summation order). The
        median is **approximate**: it is estimated from a reservoir sample of
        `reservoir_size` values per variable, and its error bound (from the
        Dvoretzky-Kiefer-Wolfowitz inequality, with 95% confidence) is
        recorded in `statistics_snapshot_.error_bounds`.

        For a fixed `random_state`, the same sequence of chunks always gives
        the same result. No multiprocessing is used, so the number of
        processes does not affect the result.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The next chunk of the training dataset.

        y: pandas series or None, default=None
            y is not needed in this imputation. You can pass None or y.

        reservoir_size: int, default=1024
            Number of values kept per variable to estimate the median. Only
            used on the first call. Bigger reservoirs give tighter error
            bounds; memory cost is reservoir_size floats per variable,
            independent of the data size.
        """
        X = check_X(X)

        accumulator = getattr(self, "_stats_accumulator_", None)
        if accumulator is None:
            # block partial_fit after a full fit when statistics are frozen
            self._guard_against_refit()

            if self.variables is None:
                variables_ = find_numerical_variables(X, self.return_empty)
            else:
                variables_ = check_numerical_variables(X, self.variables)

            self.variables_ = variables_
            self._stats_accumulator_ = ChunkedStatisticsAccumulator(
                variables_,
                reservoir_size=reservoir_size,
                random_state=self.random_state,
            )
            self._get_feature_names_in(X)
        else:
            _check_X_matches_training_df(X, self.n_features_in_)
            X = X[self.feature_names_in_]

        accumulator = self._stats_accumulator_
        accumulator.update(X[self.variables_])

        if self.imputation_method == "mean":
            self.imputer_dict_ = {v: accumulator.mean(v) for v in self.variables_}
            exact = {"mean": True}
            error_bounds = {}
            confidence = None
        else:
            estimates, bounds = accumulator.quantiles_with_bounds((0.5,))
            self.imputer_dict_ = {v: estimates[v]["q0.5"] for v in self.variables_}
            exact = {"median": False}
            error_bounds = {v: {"median": bounds[v]["q0.5"]} for v in self.variables_}
            confidence = DEFAULT_CONFIDENCE

        # chunked fits always record their provenance: the exactness of the
        # statistics is part of the contract of partial_fit()
        self._record_fit_statistics(
            None,
            statistics=self._get_fitted_statistics(),
            scope="chunked",
            exact=exact,
            error_bounds=error_bounds,
            error_bound_confidence=confidence,
            n_chunks=accumulator.n_chunks,
            random_state=self.random_state,
            distribution_summary={
                v: accumulator.distribution_summary(v) for v in self.variables_
            },
            n_samples=accumulator.n_rows,
            force=True,
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replace missing data with the learned parameters.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_new: pandas dataframe of shape = [n_samples, n_features]
            The dataframe without missing values in the selected variables.
        """
        # fail loudly if the frozen statistics are missing; never fall back
        # to default values
        self._check_statistics_available()
        return super().transform(X)

    def _get_fitted_statistics(self):
        """Statistics learned during fit, as {variable: {name: value}}."""
        return {
            v: {self.imputation_method: float(self.imputer_dict_[v])}
            for v in self.variables_
        }


# TODO: remove in version 2.1.0
class MeanMedianImputer(MeanImputer):
    def __init__(
        self,
        imputation_method: str = "median",
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_empty: bool = False,
    ) -> None:
        warnings.warn(
            "MeanMedianImputer was deprecated in favour of MeanImputer in version "
            "2.0.0 and will be removed in version 2.1.0. To silence this warning, "
            "use MeanImputer instead.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(
            imputation_method=imputation_method,
            variables=variables,
            return_empty=return_empty,
        )
