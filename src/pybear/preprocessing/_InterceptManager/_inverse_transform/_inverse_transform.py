# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
from .._type_aliases import (
    InternalDataContainer,
    RemovedColumnsType,
    FeatureNamesInType
)

import numpy as np
import pandas as pd
import scipy.sparse as ss
import polars as pl



def _inverse_transform(
    X: InternalDataContainer,
    _removed_columns: RemovedColumnsType,
    _feature_names_in: Union[FeatureNamesInType, None]
) -> InternalDataContainer:

    """
    Revert transformed data back to its original state. IM cannot account
    for any nan-like values that may have been in the original data.


    Parameters
    ----------
    X :
        array-like of shape (n_samples, n_transformed_features) - A
        transformed data set. Any appended intercept column (via
        a :param: 'keep' dictionary) needs to be removed before coming
        into this module. Data must be received as numpy ndarray, pandas
        dataframe, polars dataframe, or scipy csc only.
    _removed_columns:
        RemovedColumnsType - the keys are the indices of constant columns
        removed from the original data, indexed by their column location
        in the original data; the values are the constant value that was
        in that column.
    _feature_names_in:
        Union[npt.NDArray[str], None] - the feature names found during
        fitting if X was passed in a container with a header.


    Returns
    -------
    -
        X_tr: array-like of shape (n_samples, n_features) - Transformed
        data reverted back to its original state.

    """


    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    assert isinstance(X,
        (np.ndarray, pd.core.frame.DataFrame, pl.DataFrame, ss.csc_matrix,
         ss.csc_array)
    )
    assert isinstance(_removed_columns, dict)
    assert all(map(isinstance, _removed_columns, (int for _ in _removed_columns)))
    if _feature_names_in is not None:
        assert isinstance(_feature_names_in, np.ndarray)
        assert all(
            map(isinstance, _feature_names_in, (str for _ in _feature_names_in))
        )
    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # retain what the original format was
    # if data is a pd/pl df, convert to numpy
    # if data is a scipy sparse convert to csc
    _og_X_format = type(X)

    if isinstance(X, (pd.core.frame.DataFrame, pl.DataFrame)):
        # remove any header that may be on this df, feature_names_in
        # will go on if available, otherwise container default header
        X = X.to_numpy()

    # must do this from left to right!
    # use the _removed_columns dict to insert columns with the original
    # constant values
    if isinstance(X, np.ndarray):   # pd/pl was converted to np
        for _rmv_idx, _value in _removed_columns.items():  # this was sorted above
            X = np.insert(
                X,
                _rmv_idx,
                np.full((X.shape[0],), _value),
                axis=1
            )
    elif isinstance(X, (ss._csc.csc_array, ss._csc.csc_matrix)):
        for _rmv_idx, _value in _removed_columns.items():  # this was sorted above
            X = ss.hstack(
                (
                    X[:, :_rmv_idx],
                    ss.csc_matrix(np.full((X.shape[0], 1), _value)),
                    X[:, _rmv_idx:]
                ),
                format="csc",
                dtype=X.dtype
            )
    else:
        raise Exception


    X = _og_X_format(X) if _og_X_format is not np.ndarray else X

    # if was a dataframe and feature names are available, reattach
    if _feature_names_in is not None \
            and _og_X_format in [pd.core.frame.DataFrame, pl.DataFrame]:
        X.columns = _feature_names_in


    return X





