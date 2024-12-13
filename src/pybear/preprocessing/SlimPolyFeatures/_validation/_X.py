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
        n_features) - the data to undergo polynomial expansion.


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


    try:
        # block non-numeric
        print(f'pizza print b4 {_X=}')
        from ....utilities import nan_mask_numerical, nan_mask
        # pizza this is what we need
        # _X[nan_mask(_X)] = np.nan
        # _X.astype(np.float64)





        # if isinstance(_X, np.ndarray):
        #     _X.astype(np.float64)
        # elif isinstance(_X, pd.core.frame.DataFrame):
        #     # pizza, as of 24_12_13_12_00_00, need to convert pd nan-likes to np.nan,
        #     # must use nan_mask not nan_mask_numerical. .astype(np.float64) is
        #     # trippin when having to convert pd nan-likes to float.
        #     _X[nan_mask(_X)] = np.nan
        #     _X.astype(np.float64)
        # elif hasattr(_X, 'toarray'):
        #     _X[nan_mask(_X)] = np.nan
        #     _X.astype(np.float64)
        #     # _X_data = np.array(_X.data)
        #     # _X_data[nan_mask(_X_data)] = np.nan
        #     # _X_data.astype(np.float64)
        # else:
        #     raise Exception
        print(f'pizza print after {_X=}')

        # pizza experiment, pd nan-likes blowing up on .astype(np.float64)
        # try:
        #     _X.astype(np.float64)
        # except:

    except:
        raise ValueError(f"X can only contain numeric datatypes")






    if _X.shape[0] < 1:
        raise ValueError(
            f"'X' must be a valid 2 dimensional numpy ndarray, pandas dataframe, "
            f"or scipy sparce matrix or array, with at least 2 columns and 1 "
            f"example."
    )


