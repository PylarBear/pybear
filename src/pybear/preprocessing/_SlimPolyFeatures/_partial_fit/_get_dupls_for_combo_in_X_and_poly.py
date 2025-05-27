# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
import numpy.typing as npt
from .._type_aliases import InternalDataContainer

import itertools
import numbers

import joblib
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as ss

from .._partial_fit._parallel_column_comparer import _parallel_column_comparer
from .._partial_fit._columns_getter import _columns_getter



def _get_dupls_for_combo_in_X_and_poly(
    _COLUMN: npt.NDArray[any],
    _X: InternalDataContainer,
    _POLY_CSC: Union[ss.csc_array, ss.csc_matrix],
    _min_degree: numbers.Integral,  # pizza added to deal with X dupls influencing output when min_degree > 1
    _equal_nan: bool,
    _rtol: numbers.Real,
    _atol: numbers.Real,
    _n_jobs: Union[numbers.Integral, None]
) -> list[bool]:


    """
    Scan a polynomial column generated as a product of a combination of
    columns from X across X itself and POLY, looking for duplicates.
    There are no duplicates in POLY_CSC (if there are it is an algorithm
    failure,) but may be in X (which is degenerate but is understood and
    handled by SPF.)


    Parameters
    ----------
    _COLUMN:
        npt.NDArray[any] - A column that is the product of a polynomial
        combo from X.
    _X:
        Union[np.ndarray, pd.DataFrame, scipy.sparse] of shape
        (n_samples, n_features) - the data to undergo polynomial
        expansion. _X will be passed to _columns_getter which allows
        ndarray, pd.DataFrame, and all scipy sparse except coo
        matrix/array, dia matrix/array, or bsr matrix/array. _X should
        be conditioned for this when passed here.
    _POLY_CSC:
        Union[ss.csc_array, ss.csc_matrix] of shape (n_samples,
        n_poly_features) - the in-progress deduplicated polynomial
        features found during expansion.
    _equal_nan:
        bool - how to handle nan-like values when checking for equality.
        See the detailed explanation in the SPF main module.
    _rtol:
        numbers.Real - The relative difference tolerance for equality.
        See numpy.allclose.
    _atol:
        numbers.Real - The absolute tolerance parameter for equality.
        See numpy.allclose.
    _n_jobs:
        Union[numbers.Integral, None] - The number of joblib Parallel
        jobs to use when looking for duplicate columns across X and POLY.


    Return
    ------
    -
        _all_dupls: list[bool] - 1D list of shape (n_X_features +
        n_POLY_features,). Column-by-column True/False stating equality
        of _COLUMN to that particular slot's associated column in X or
        POLY.  Boolean True indicates the column in X or POLY is
        identical to _COLUMN.

    """


    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # _COLUMN is always ndarray, this is determined by how the output of
    # _columns_getter() is multiplied through at the top of SPF.partial_fit().
    assert isinstance(_COLUMN, np.ndarray)
    assert len(_COLUMN.shape) == 1   # a 1D vector

    # _X passes through SPF.partial_fit() into this as given
    assert isinstance(_X, (np.ndarray, (pd.core.frame.DataFrame, pl.DataFrame))) or \
           hasattr(_X, 'toarray')
    assert not isinstance(_X,
        (ss.coo_matrix, ss.coo_array, ss.dia_matrix,
        ss.dia_array, ss.bsr_matrix, ss.bsr_array)
    )
    assert _X.shape[1] >= 1   # must always have 1 or more features

    #  _POLY_CSC is constructed as a ss csc_array in SPF.partial_fit()
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


    args = (_COLUMN, _rtol, _atol, _equal_nan)


    _n_cols = 200


    # look for duplicates in X
    # can use _parallel_column_comparer since _columns_getter turns ss to dense
    # there can be more than one hit for duplicates in X
    _X_dupls = []
    # if _min_degree == 1:   # pizza
    if _X.shape[1] < 2 * _n_cols:
        for c_idx in range(_X.shape[1]):
            _X_dupls.append(
                _parallel_column_comparer(_columns_getter(_X, c_idx),  *args)[0]
            )
    else:
        with joblib.parallel_config(prefer='processes', n_jobs=_n_jobs):
            _X_dupls = joblib.Parallel(return_as='list')(
                joblib.delayed(_parallel_column_comparer)(
                _columns_getter(
                    _X,
                    tuple(range(i, min(i + _n_cols, _X.shape[1])))
                ),
                *args
                ) for i in range(0, _X.shape[1], _n_cols)
            )

        _X_dupls = list(itertools.chain(*_X_dupls))
    # else:
    #     _X_dupls = [False] * _X.shape[1]




    # look for duplicates in POLY
    # can use _parallel_column_comparer since _columns_getter turns ss to dense
    # there *cannot* be more than one hit for duplicates in POLY
    _poly_dupls = []
    if _POLY_CSC.shape[1] < 2 * _n_cols:
        for c_idx in range(_POLY_CSC.shape[1]):
            _poly_dupls.append(
                _parallel_column_comparer(_columns_getter(_POLY_CSC, c_idx), *args)[0]
            )
    else:
        with joblib.parallel_config(prefer='processes', n_jobs=_n_jobs):
            _poly_dupls = joblib.Parallel(return_as='list')(
                joblib.delayed(_parallel_column_comparer)(
                    _columns_getter(
                        _POLY_CSC,
                        tuple(range(i, min(i + _n_cols, _POLY_CSC.shape[1])))
                    ),
                    *args
                ) for i in range(0, _POLY_CSC.shape[1], _n_cols)
            )

        _poly_dupls = list(itertools.chain(*_poly_dupls))

    # del POLY_RANGE




    # there cannot be a hit in both X and POLY, if so, this is a serious
    # algorithm failure in that a combo determined to be a duplicate of
    # a column in X has been put into POLY.
    assert any(_X_dupls) + any(_poly_dupls) in [0, 1]

    _all_dupls = _X_dupls + _poly_dupls


    return _all_dupls











