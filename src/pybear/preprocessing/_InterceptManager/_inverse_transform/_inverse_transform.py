# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
import numpy.typing as npt
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
    _feature_names_in: FeatureNamesInType
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
        dataframe, or scipy csc only.
    _removed_columns:
        RemovedColumnsType - the keys are the indices of constant columns
        removed from the original data, indexed by their column location
        in the original data; the values are the constant value that was
        in that column.
    _feature_names_in:
        Union[npt.NDArray[str], None] - the feature names found during
        fitting if X was passed as a dataframe with a header.


    Returns
    -------
    -
        X_tr: array-like of shape (n_samples, n_features) - Transformed
        data reverted back to its original state.

    """

    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
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
    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    # retain what the original format was
    # if data is a pd df, convert to numpy
    # if data is a scipy sparse convert to csc
    _og_X_format = type(X)

    if isinstance(X, (pd.core.frame.DataFrame, pl.DataFrame)):
        # remove any header that may be on this df, dont want to replicate
        # wrong headers in 'new' columns which would be exposed if
        # feature_names_in_ is not available (meaning that fit() never
        # saw a header via df, but a df has been passed to
        # inverse_transform.)
        X = X.to_numpy()


    # must do this from left to right!
    # use the _removed_columns dict to insert columns with the original
    # constant values
    if isinstance(X, np.ndarray):   # pd was converted to np
        for _rmv_idx, _value in _removed_columns.items():  # this was sorted above
            X = np.insert(X, _rmv_idx, np.full((X.shape[0],), _value), axis=1)
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


    # if was a dataframe and feature names are available, reattach
    if _og_X_format is np.ndarray:
        pass
    elif _og_X_format is pd.core.frame.DataFrame:
        X = pd.DataFrame(data=X)

        if _feature_names_in is not None:
            X.columns = _feature_names_in
    elif _og_X_format is pl.DataFrame:
        X = pl.DataFrame(data=X)

        if _feature_names_in is not None:
            X.columns = _feature_names_in
    else:
        X = _og_X_format(X)

    return X





