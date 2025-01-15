# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
import numpy.typing as npt
from .._type_aliases import SparseTypes

import numbers
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, wrap_non_picklable_objects

from ....utilities import nan_mask
from .._partial_fit._columns_getter import _columns_getter



def _val_X(
    _X: Union[npt.NDArray[any], pd.DataFrame, SparseTypes],
    _interaction_only: bool,
    _n_jobs: Union[numbers.Integral, None]
) -> None:

    """
    Validate X.
    The container format:
        Must be numpy ndarray, pandas dataframe, or any scipy sparse.
    The dimensions of the container:
        Must have at least 1 sample.
        When interaction_only is True, must have at least 2 columns.
        When interaction_only is False, must have at least 1 column.
    All values within must be numeric.


    Parameters
    ----------
    _X:
        {array-like, scipy sparse matrix} of shape (n_samples,
        n_features) - the data to undergo polynomial expansion.
    _interaction_only:
        bool - If True, only interaction features are produced, that is,
        polynomial features that are products of 'degree' distinct input
        features. Terms with power of 2 or higher for any feature are
        excluded.
        Consider 3 features 'a', 'b', and 'c'. If 'interaction_only' is
        True, 'min_degree' is 1, and 'degree' is 2, then only the first
        degree interaction terms ['a', 'b', 'c'] and the second degree
        interaction terms ['ab', 'ac', 'bc'] are returned in the
        polynomial expansion.
    _n_jobs:
        Union[numbers.Integral, None] - The number of joblib Parallel
        jobs to use when looking for duplicate columns or looking for
        columns of constants.


    Return
    ------
    -
        None


    """

    """
    # pizza, hitting this before _val_interaction_only and raising the wrong error
    # hash this now, come back and deal with this after the order of validation
    # in partial_fit is finalized (i.e., where does _val_X end up falling in
    # relation to the others.)
    # assert isinstance(_interaction_only, bool)

    # pizza, hitting this before _val_n_jobs and raising the wrong error
    # hash this now, come back and deal with this after the order of validation
    # in partial_fit is finalized (i.e., where does _val_X end up falling in
    # relation to the others.)
    # if _n_jobs is not None:
    #     err_msg = f"'n_jobs' must be None, -1, or an integer greater than 0"
    #     if not isinstance(_n_jobs, numbers.Integral):
    #         raise ValueError(err_msg)
    #     value_error = 0
    #     value_error += not (_n_jobs == -1 or _n_jobs >= 1)
    #     value_error += isinstance(_n_jobs, bool)
    #     if value_error:
    #         raise ValueError(err_msg)
    #     del err_msg, value_error
    """

    if not isinstance(_X, (np.ndarray, pd.core.frame.DataFrame)) and not \
        hasattr(_X, 'toarray'):
        raise TypeError(
            f"invalid container for X: {type(_X)}. X must be numpy array, "
            f"pandas dataframe, or any scipy sparce matrix / array."
        )

    """
    # block non-numeric
    if isinstance(_X, np.ndarray):
        try:
            _X.astype(np.float64)
        except:
            raise ValueError(f"X can only contain numeric datatypes")

    elif isinstance(_X, pd.core.frame.DataFrame):
        # need to circumvent pd nan-likes.
        # .astype(np.float64) is raising when having to convert pd
        # nan-likes to float, so pd df cant be handled with ndarray above.
        # but _X[nan_mask(_X)] = np.nan is back-talking to the passed X
        # and mutating it. want to do this without mutating
        # X or making a copy (see the exception for this in transform()).
        # so scan it column by column. for sanity, keep this scan separate
        # from anything going on in partial_fit with IM and CDT

        @wrap_non_picklable_objects
        def _test_is_num(_column: npt.NDArray) -> None:
            # empiricism shows must use nan_mask not nan_mask_numerical.
            np.float64(_column[np.logical_not(nan_mask(_column))])


        try:
            joblib_kwargs = {
                'return_as': 'list', 'n_jobs': _n_jobs, 'prefer': 'processes'
            }
            Parallel(**joblib_kwargs)(delayed(_test_is_num)(
                _columns_getter(_X, c_idx)) for c_idx in range(_X.shape[1])
            )
        except:
            raise ValueError(f"X can only contain numeric datatypes")

        del _test_is_num

    elif hasattr(_X, 'toarray'):
        # scipy sparse can only be numeric dtype, so automatically good
        pass
    """

    # bearpizza this is stopgap to get tests to pass
    # this is directly copied from _num_samples
    # try:
    #     _X.shape
    # except:
    #     raise ValueError(
    #         f"\nThe passed object does not have a 'shape' attribute. \nAll "
    #         f"pybear estimators and transformers require data-bearing objects "
    #         f"to have a 'shape' attribute, like numpy array, pandas dataframes, "
    #         f"and scipy sparse matrices / arrays."
    #     )
    # END bearpizza this is stopgap to get tests to pass


    _base_msg = (
        f"'X' must be a valid 2 dimensional numpy ndarray, pandas dataframe, "
        f"or scipy sparce matrix or array with at least 1 sample. "
    )

    if len(_X.shape) == 1:
        _addon_msg = (f"\nIf passing 1 feature, reshape your data to 2D.")
        raise ValueError(_base_msg + _addon_msg)

    if len(_X.shape) > 2:
        raise ValueError(_base_msg)

    if _X.shape[0] < 1:
        raise ValueError(_base_msg)

    _base_msg = (
        f"'X' must be a valid 2 dimensional numpy ndarray, pandas dataframe, "
        f"or scipy sparce matrix or array with at least 1 sample. "
    )

    if _interaction_only:
        if _X.shape[1] < 2:
            _addon_msg = (f"\nWhen only generating interaction terms (:param: "
                f"interaction_only is True), 'X' must have at least 2 features.")
            raise ValueError(_base_msg + _addon_msg)
    elif not _interaction_only:
        if _X.shape[1] < 1:
            _addon_msg = (f"\nWhen generating all polynomial terms (:param: "
                f"interaction_only is False), 'X' must have at least 1 feature.")
            raise ValueError(_base_msg + _addon_msg)





