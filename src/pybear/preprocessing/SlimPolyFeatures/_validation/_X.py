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
    _X: Union[npt.NDArray[any], pd.DataFrame, SparseTypes],
    _interaction_only: bool
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
    _interaction_only:
        bool - pizza say something!

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



    # block non-numeric
    if isinstance(_X, np.ndarray):
        try:
            _X.astype(np.float64)
        except:
            raise ValueError(f"X can only contain numeric datatypes")

    elif isinstance(_X, pd.core.frame.DataFrame):
        # pizza, as of 24_12_13_12_00_00, need to convert pd nan-likes to np.nan,
        # empiricism shows must use nan_mask not nan_mask_numerical.
        # .astype(np.float64) is trippin when having to convert pd nan-likes to float.
        # but _X[nan_mask(_X)] = np.nan is back-talking to the passed X
        # and mutating it. want to do this without mutating X or making a copy.
        # so scan it columns by column. for sanity, keep this scan separate from
        # anything going on in partial_fit with IM and CDT
        from ....utilities import nan_mask_numerical, nan_mask
        from .._partial_fit._columns_getter import _columns_getter
        from joblib import Parallel, delayed, wrap_non_picklable_objects

        @wrap_non_picklable_objects
        def _test_is_num(_column: npt.NDArray) -> None:
            np.float64(_column[np.logical_not(nan_mask(_column))])

        # pizza if this all works out, pass n_jobs to this module
        try:
            joblib_kwargs = {'return_as': 'list', 'n_jobs': -1, 'prefer': 'processes'}
            Parallel(**joblib_kwargs)(delayed(_test_is_num)(_columns_getter(_X, c_idx)) for c_idx in range(_X.shape[1]))
        except:
            raise ValueError(f"X can only contain numeric datatypes")

        del _test_is_num

    elif hasattr(_X, 'toarray'):
        # scipy sparse can only be numeric dtype, so automatically good
        pass


    _base_msg = (f"'X' must be a valid 2 dimensional numpy ndarray, pandas "
                 f"dataframe, or scipy sparce matrix or array with at least 1 sample. ")

    if _X.shape[0] < 1:
        raise ValueError(_base_msg)

    if _interaction_only:
        if _X.shape[1] < 2:
            _addon_msg = (f"\nWhen only generating interaction terms (:param: "
                f"interaction_only is True), 'X' must have at least 2 features.")
            raise ValueError(_base_msg + _addon_msg)
    elif not _interaction_only:
        _addon_msg = (f"\nWhen generating all polynomial terms (:param: "
            f"interaction_only is False), 'X' must have at least 1 feature.")
        raise ValueError(_base_msg + _addon_msg)





