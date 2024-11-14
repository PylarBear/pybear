# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
import numpy.typing as npt
from .._type_aliases import SparseTypes

import pandas as pd
import scipy.sparse as ss




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

    # sklearn _validate_data & check_array are not catching this
    if _X is None:
        raise TypeError(_err_msg)


    # sklearn _validate_data & check_array are not catching this
    if len(_X.shape) != 2 or _X.shape[1] == 1:
        raise ValueError


    # sklearn _validate_data & check_array are not catching this.
    # what appears to be happening is that they are converting bsr
    # to the first allowed form in the 'accept_sparse' list.
    if isinstance(_X, (ss.bsr_array, ss.bsr_matrix)):
        raise TypeError(f"X cannot be a scipy BSR matrix / array")









