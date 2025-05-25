# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
import numpy.typing as npt

import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as ss






def _transform(
    _X: Union[npt.NDArray, pd.core.frame.DataFrame, ss.csc_matrix, ss.csc_array],
    _column_mask: npt.NDArray[bool]
) -> Union[npt.NDArray, pd.core.frame.DataFrame, ss.csc_matrix, ss.csc_array]:

    """
    Remove the duplicate columns from X as indicated in the _column_mask
    vector.


    Parameters
    ----------
    _X:
        {array-like, scipy sparse csc} of shape (n_samples,
        n_features) - The data to be deduplicated. The container can only
        be numpy ndarray, pandas dataframe, or scipy sparse csc.
    _column_mask:
        NDArray[bool] - A boolean vector of shape (n_features,) that
        indicates which columns to keep (True) and which columns to
        delete (False).


    Return
    ------
    -
        _X: {array-like, scipy sparse csc} of shape (n_samples,
            n_features - n_features_removed) - The deduplicated data.

    """

    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    assert isinstance(
        _X,
        (np.ndarray, pd.core.frame.DataFrame, pl.DataFrame, ss.csc_matrix, ss.csc_array)
    )
    assert isinstance(_column_mask, np.ndarray)
    assert _column_mask.dtype == bool

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    if isinstance(_X, np.ndarray):
        return _X[:, _column_mask]
    elif isinstance(_X, pd.core.frame.DataFrame):
        return _X.loc[:, _column_mask]
    elif isinstance(_X, pl.DataFrame):
        return _X[:, _column_mask]
    elif isinstance(_X, (ss.csc_matrix, ss.csc_array)):
        return _X[:, _column_mask]
    else:
        raise Exception





