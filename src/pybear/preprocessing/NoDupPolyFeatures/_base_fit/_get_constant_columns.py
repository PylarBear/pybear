# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy as np
import numpy.typing as npt
from .._type_aliases import DataType


def _get_constant_columns(
    X: DataType
) -> npt.NDArray[np.int32]:

    """
    Determine the columns of X that are constants (all values in the
    column are the same.


    Parameters
    ----------
    X:
        {array-like, scipy sparse matrix} of shape (n_samples, n_features).
        The data.

    Return
    ------
    -
        NDArray[int]: indices of constant columns

    """


    if hasattr(X, 'toarray'):
        # is scipy sparse
        # pizza, finish this

    else:
        _variances = np.nanstd(X, axis=0)


    return np.arange(X.shape[1])[_variances == 0].astype(np.int32)






