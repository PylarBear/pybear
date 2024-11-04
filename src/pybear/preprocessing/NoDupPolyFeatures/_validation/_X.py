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

    _err_msg = (
        f"'X' must be a valid 2 dimensional numpy ndarray, pandas dataframe, "
        f"or scipy sparce matrix or array, with at least 2 columns and 1 "
        f"example."
    )

    # sklearn _validate_data is not catching this
    if _X is None:
        raise TypeError(_err_msg)


    # rejects non-numeric, does not accept nan-likes
    # BaseEstimator _validate_data is catching nans with force_all_finite = False,
    # but keep this checking nans in case _validate_data should ever change
    try:
        _X.astype(np.uint8)
        # this kills two birds, both non-numeric and nan cannot convert to int8
    except:
        raise ValueError(
            f"data must be numeric and cannot have nan-like values"
        )






