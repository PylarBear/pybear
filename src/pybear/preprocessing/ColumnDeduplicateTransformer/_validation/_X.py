# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
import numpy.typing as npt
from .._type_aliases import SparseContainer

import numpy as np
import pandas as pd
import scipy.sparse as ss




def _val_X(
    _X: Union[npt.NDArray[any], pd.DataFrame, SparseContainer]
) -> None:

    """
    Validate the container type of the data. Cannot be None and cannot
    be a scipy BSR matrix / array. Otherwise, X can be a numpy ndarray,
    a pandas dataframe, or any other scipy sparse matrix / array.

    All other validation of the data is handled by the validate_data
    function at fit, transform, and inverse_transform.


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
            f"pandas dataframe, or any scipy sparce matrix / array except "
            f"BSR."
        )


    if isinstance(_X, (ss.bsr_array, ss.bsr_matrix)):
        raise TypeError(f"X cannot be a scipy BSR matrix / array.")










