# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
import numpy.typing as npt
from .._type_aliases import SparseTypes

import numpy as np
import pandas as pd



def _val_X(
    _X: Union[npt.NDArray[any], pd.DataFrame, SparseTypes]
) -> None:

    """
    Validate the dimensions of the data to be deduplicated. Cannot be
    None and must have at least 2 columns.

    All other validation of the data is handled by the _validate_data
    function of the sklearn BaseEstimator mixin at fitting and tranform.


    Parameters
    ----------
    _X:
        {array-like, scipy sparse matrix} of shape (n_samples,
        n_features) - the data to be deduplicated.


    Return
    ------
    -
        None


    """


    # sklearn _validate_data & check_array are not catching dask arrays & dfs.
    if not isinstance(_X, (np.ndarray, pd.core.frame.DataFrame)) and not \
        hasattr(_X, 'toarray'):
        raise TypeError(
            f"invalid container for X: {type(_X)}. X must be numpy array, "
            f"pandas dataframe, or any scipy sparce matrix / array."
        )

    # pizza see what _validate_data can do
    # rejects non-numeric, does not accept nan-likes (pizza we need to accept nans!)
    # BaseEstimator _validate_data is catching nans with force_all_finite = False,
    # but keep this checking nans in case _validate_data should ever change
    try:
        _X.astype(np.float64)
        # this kills two birds, both non-numeric and nan cannot convert to int8
    except:
        raise ValueError(f"X can only contain numeric datatypes")






    if _X.shape[0] < 1:
        raise ValueError(
            f"'X' must be a valid 2 dimensional numpy ndarray, pandas dataframe, "
            f"or scipy sparce matrix or array, with at least 2 columns and 1 "
            f"example."
    )


