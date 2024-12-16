# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
import numpy.typing as npt

import numbers

from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import scipy.sparse as ss

from .._partial_fit._parallel_column_comparer import _parallel_column_comparer
from .._partial_fit._columns_getter import _columns_getter




def _get_dupls_for_combo_in_X_and_poly(
    _COLUMN: npt.NDArray[any],
    _X: Union[ss.csc_array, ss.csc_matrix],
    _POLY_CSC: Union[ss.csc_array, ss.csc_matrix],
    _equal_nan: bool,
    _rtol: numbers.Real,
    _atol: numbers.Real,
    _n_jobs: Union[numbers.Integral, None]
) -> list[bool]:

    # there are no duplicates in POLY_CSC,
    # but may be in X

    # look_for_duplicates_in_X
    # _look_for_duplicates_in_X needs to return the idx in X that the
    # combo matches so both X idx and combo can be put into the
    # _duplicates dict
    # cannot overwrite self.duplicates_! may have previous fits in it

    """

    Parameters
    ----------
    _COLUMN:
        npt.NDArray[any] - The column that is the product of the
        polynomial combo
    _X:
        {array-like, scipy sparse} of shape (n_samples, n_features) - the
        data to undergo polynomial expansion.
    _POLY_CSC:
        {ss.csc_array} of shape (n_samples, n_poly_features) - the
        in-progress non-duplicate polynomial features found during
        expansion.
    _equal_nan:
        bool -   
    _rtol:
        numbers.Real - The relative difference tolerance for
            equality. See numpy.allclose.
    _atol:
        numbers.Real - The absolute tolerance parameter for .
            equality. See numpy.allclose.
    _n_jobs: Union[numbers.Integral, None]


    Return
    ------
    -
        _all_dupls: list[bool] - list of shape (n_X_features + n_POLY_features,).
        Boolean True indicates the column is identical to _COLUMN.


    """


    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # _COLUMNS is always coming out of _columns_getter() as np.ndarray, but leave
    # this flexible in case we ever end up returning from _columns_getter() as
    # passed

    # pizza revisit this, currently _COLUMN is returned as ndarray and multiplied thru as ndarray
    # assert isinstance(_COLUMN, (np.ndarray, pd.core.frame.DataFrame, ss.csc_array, ss.csc_matrix))
    assert isinstance(_COLUMN, np.ndarray)
    assert len(_COLUMN.shape) == 1
    # pizza revisit this, currently at the top of SPF._partial_fit() setting _X
    # to ss.csc, but as of 24_12_16 this not the case allows np, pd, and ss
    assert isinstance(_X, (np.ndarray, pd.core.frame.DataFrame)) or hasattr(_X, 'toarray')
    # assert isinstance(_X, (ss.csc_array, ss.csc_matrix))
    assert _X.shape[1] >= 2
    # pizza revisit this, currently _POLY_CSC is constructed as a ss csc_array
    assert isinstance(_POLY_CSC, (ss.csc_array, ss.csc_matrix))
    assert len(_COLUMN) == _X.shape[0] == _POLY_CSC.shape[0]
    assert isinstance(_equal_nan, bool)
    assert isinstance(_rtol, numbers.Real)
    assert not isinstance(_rtol, bool)
    assert _rtol >= 0
    assert isinstance(_atol, numbers.Real)
    assert not isinstance(_atol, bool)
    assert _atol >= 0
    assert isinstance(_n_jobs, (numbers.Integral, type(None)))
    assert not isinstance(_n_jobs, bool)
    assert _n_jobs is None or (_n_jobs >= -1 and _n_jobs != 0)




    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    joblib_kwargs = {
        'prefer': 'processes', 'n_jobs': _n_jobs, 'return_as': 'list'
    }
    args = (_COLUMN, _rtol, _atol, _equal_nan)

    # def _parallel_column_comparer(
    #     _column1: npt.NDArray[any],
    #     _column2: npt.NDArray[any],
    #     _rtol: numbers.Real,
    #     _atol: numbers.Real,
    #     _equal_nan: bool
    # ) -> bool:


    # must use _parallel_column_comparer since _columns_getter explodes ss out to dense now
    # there can be more than one hit for duplicates in X
    _X_dupls = Parallel(**joblib_kwargs)(
        delayed(_parallel_column_comparer)(_columns_getter(_X, c_idx).ravel(), *args) for c_idx in range(_X.shape[1])
    )

    # must use _parallel_column_comparer since _columns_getter explodes ss out to dense now
    _poly_dupls = Parallel(**joblib_kwargs)(
        delayed(_parallel_column_comparer)(_columns_getter(_POLY_CSC, c_idx).ravel(), *args) for c_idx in range(_POLY_CSC.shape[1])
    )

    # if there is a duplicate in X, there cannot be a duplicate in poly.
    if any(_X_dupls):
        assert not any(_poly_dupls)
    # pizza
    # if there is no duplicate in X, there can only be zero or one duplicate in poly.
    # elif not any(_X_dupls):
    #     _poly_dupls could be anything

    _all_dupls = _X_dupls + _poly_dupls


    return _all_dupls











