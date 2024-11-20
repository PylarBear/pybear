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
    pizza
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

    _err_msg = (
        f"'X' must be a valid 2 dimensional numpy ndarray, pandas dataframe, "
        f"or scipy sparce matrix or array, with at least 2 columns and 1 "
        f"example."
    )

    # pizza
    # sklearn _validate_data is not catching this
    if _X is None:
        raise TypeError(_err_msg)
    #
    #
    # # sklearn _validate_data is not catching this
    # if len(_X.shape) != 2:
    #     raise UnicodeError


    # sklearn _validate_data & check_array are not catching dask arrays & dfs.
    if not isinstance(_X, (np.ndarray, pd.core.frame.DataFrame)) and not \
        hasattr(_X, 'toarray'):
        raise TypeError(
            f"invalid container for X: {type(_X)}. X must be numpy array, "
            f"pandas dataframe, or any scipy sparce matrix / array except "
            f"BSR."
        )



