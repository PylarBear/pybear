# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy as np
import numpy.typing as npt
import pandas as pd
from joblib import Parallel

from typing_extensions import Union

from pybear.preprocessing.NoDupPolyFeatures._type_aliases import DataType

from sklearn.utils.sparsefuncs import min_max_axis


# pizza, this does a conversion to csc, may want to standardize ss format early in NoDup
# pizza, this converts a pd to np, may want to standardize pd format early in NoDup


def _get_constant_columns(
    _X: DataType,
    _equal_nan: bool,
    _rtol: float,
    _atol: float,
    _as_indices: bool=True,
    _n_jobs: Union[int, None]=None
) -> npt.NDArray[Union[np.int32, bool]]:

    """
    Determine the columns of X that are constants (all values in the
    column are the same.)


    Parameters
    ----------
    _X:
        {array-like, scipy sparse matrix} of shape (n_samples, n_features).
        The data.
    _equal_nan:
        bool -
    _rtol:
        float -
    _atol:
        float -
    _as_indices:
        bool - True: return as indices; False: return as boolean vector of
            shape (n_features,)
    _n_jobs:
        Union[int, None], default=None. Pizza finish.

    Return
    ------
    -
        NDArray[Union[int, bool]]: indices of constant columns

    """

    # _X must be np array, pddf, or scipy sparse
    assert isinstance(_X, (np.ndarray, pd.core.frame.DataFrame)) or \
        hasattr(_X, "toarray")

    assert isinstance(_as_indices, bool)

    # because we are using allclose and making allowance for user-defined
    # handling of nans, we cant use np.ptp for a fast diagnosis on np and
    # pd.
    # sklearn handling of nans in the sparse matrix functions (min_max_axis,
    # _sparse_nan_min_max, et al) is insufficient for the pybear handling
    # of nans. this means we need to iterate over the sparse matrices and
    # apply pybear nan_mask while looking for constants.



    joblib_kwargs = {'prefer':'processes', 'return_as':'list', 'n_jobs':_n_jobs}
    # out = Parallel(**joblib_kwargs)(_pizza_function(what?!?!) for what in range(X.shape[1]))






    """
    pizza, old code pre for loop
    # since only looking for zero-variance columns, just use peak-to-peak
    # to find them
    if hasattr(_X, "toarray"):  # sparse matrix
        # min_max_axis requires CSR or CSC format
        mins, maxes = min_max_axis(_X.tocsc(), ignore_nan=False, axis=0)   # pizza
        peak_to_peaks = maxes - mins
        print(f'pizza test {peak_to_peaks=}')
    elif isinstance(_X, np.ndarray):
        # pizza
        # peak_to_peaks = np.nanmax(X, axis=0) - np.nanmin(X, axis=0)
        peak_to_peaks = np.ptp(_X, axis=0)
    elif isinstance(_X, pd.core.frame.DataFrame):
        # pizza
        # peak_to_peaks = np.nanmax(X.to_numpy(), axis=0) - np.nanmin(X.to_numpy(), axis=0)
        peak_to_peaks = np.ptp(_X.to_numpy(), axis=0)   # pizza

    if _as_indices:
        return np.arange(_X.shape[1])[peak_to_peaks == 0].astype(np.int32)
    elif not _as_indices:
        return (peak_to_peaks == 0).astype(bool)
    """




