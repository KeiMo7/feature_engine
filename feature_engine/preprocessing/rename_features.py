from typing import Dict, List, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import check_X
from feature_engine.tags import _return_tags


class RenameFeatures(TransformerMixin, BaseEstimator):
    """
    RenameFeatures() renames variables in the dataframe according to a
    user-supplied mapping.

    Unlike an ad-hoc ``pandas.DataFrame.rename`` call, this transformer records
    the rename mapping in the fitted attribute ``feature_renames_``. The
    feature snapshot machinery uses that attribute to recognise the output as
    the *same* feature under a new name, instead of a removed feature plus a
    new feature.

    More details in the :ref:`User Guide <rename_features>`.

    Parameters
    ----------
    rename_dict: dict
        Dictionary mapping current variable names to new names, e.g.
        ``{"old_name": "new_name"}``.

    Attributes
    ----------
    feature_names_in_:
        List with the names of the variables seen during `fit`.

    feature_renames_:
        The validated rename mapping. Read by the snapshot machinery to keep
        feature identity stable across renames.

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        Validate the rename mapping against the input dataframe.

    fit_transform:
        Fit to data, then transform it.

    get_feature_names_out:
        Get output feature names for transformation.

    transform:
        Rename the variables.

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.preprocessing import RenameFeatures
    >>> X = pd.DataFrame(dict(x1=[1, 2, 3], x2=[4, 5, 6]))
    >>> rf = RenameFeatures(rename_dict={"x1": "age"})
    >>> rf.fit_transform(X)
       age  x2
    0    1   4
    1    2   5
    2    3   6
    """

    def __init__(self, rename_dict: Dict[Union[str, int], Union[str, int]]):
        if not isinstance(rename_dict, dict) or len(rename_dict) == 0:
            raise ValueError(
                "rename_dict should be a non-empty dictionary mapping current "
                f"column names to new names. Got {rename_dict} instead."
            )
        self.rename_dict = rename_dict

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Validate the rename mapping and store the input feature names.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The input dataframe.

        y: None
            y is not needed for this transformer. You can pass y or None.
        """

        X = check_X(X)

        self.feature_names_in_: List[Union[str, int]] = X.columns.tolist()
        self.n_features_in_ = X.shape[1]

        missing = [
            old_name
            for old_name in self.rename_dict
            if old_name not in self.feature_names_in_
        ]
        if missing:
            raise ValueError(
                "The following rename sources are not columns of the dataframe "
                f"used in fit: {missing}."
            )

        targets = list(self.rename_dict.values())
        if len(set(targets)) != len(targets):
            raise ValueError(
                "rename_dict maps two or more columns to the same new name."
            )

        already_present = [
            new_name
            for old_name, new_name in self.rename_dict.items()
            if new_name in self.feature_names_in_ and new_name not in self.rename_dict
        ]
        if already_present:
            raise ValueError(
                "The following rename targets already exist as columns and would "
                f"overwrite them: {already_present}."
            )

        self.feature_renames_ = dict(self.rename_dict)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Rename the variables according to ``rename_dict``.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: pandas dataframe
            The dataframe with renamed columns.
        """

        check_is_fitted(self)
        X = check_X(X)
        return X.rename(columns=self.feature_renames_)

    def get_feature_names_out(
        self,
        input_features: List[Union[str, int]] = None,
    ) -> List[Union[str, int]]:
        """Get output feature names for transformation."""

        check_is_fitted(self)

        if input_features is not None:
            input_features = list(input_features)
            if input_features != self.feature_names_in_:
                raise ValueError("input_features is not equal to feature_names_in_.")
            names = input_features
        else:
            names = list(self.feature_names_in_)

        return [self.feature_renames_.get(name, name) for name in names]

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["_xfail_checks"][
            "check_parameters_default_constructible"
        ] = "transformer has 1 mandatory parameter"
        tags_dict["_xfail_checks"][
            "check_transformer_general"
        ] = "the transformer renames user provided columns, so the generic "
        "column-name assumptions do not apply"
        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
