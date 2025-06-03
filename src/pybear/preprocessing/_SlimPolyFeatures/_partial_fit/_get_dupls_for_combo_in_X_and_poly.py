# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
from .._type_aliases import InternalXContainer

import itertools
import math
import numbers

import joblib
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as ss

from .._partial_fit._parallel_chunk_comparer import _parallel_chunk_comparer
from .._partial_fit._columns_getter import _columns_getter

# pizza finish this

def _get_dupls_for_combo_in_X_and_poly(
    _X: InternalXContainer,
    _poly_combos: list[tuple[int, ...]],
    _min_degree: numbers.Integral,
    _equal_nan: bool,
    _rtol: numbers.Real,
    _atol: numbers.Real,
    _n_jobs: Union[numbers.Integral, None],
    _job_size: numbers.Integral
) -> list[tuple[tuple[int,...], tuple[int, ...]]]:


    """
    Scan a polynomial column generated as a product of a combination of
    columns from X across X itself and POLY, looking for duplicates.
    There are no duplicates in POLY_CSC (if there are it is an algorithm
    failure,) but may be in X (which is degenerate but is understood and
    handled by SPF.)


    Parameters
    ----------
    _X:
        InternalXContainer of shape (n_samples, n_features) - the
        data  to undergo polynomial expansion. _X will be passed to
        _columns_getter which allows ndarray, pd.DataFrame, and all
        scipy sparse except coo
        matrix/array, dia matrix/array, or bsr matrix/array. _X should
        be conditioned for this when passed here.
    _poly_combos:
        list[tuple[int, ...]] - the combinations of columns from _X to
        use to build the polynomial columns.
    _min_degree:
        numbers.Integral - the minimum degree of polynomial terms to
        return in the output.
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
    _job_size:
        numbers.Integral - The number of columns to send to a joblib job.
        Must be an integer greater than or equal to 2.


    Return
    ------
    -
        _all_dupls: list[bool] - 1D list of tuples, each tuple holding
        two groups of indices. Each group of indices indicate column
        indices from _X that produce a duplicate column.

    """


    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    assert isinstance(_X,
        (np.ndarray, pd.core.frame.DataFrame, pl.DataFrame, ss.csc_array,
        ss.csc_matrix)
    )
    assert _X.shape[1] >= 1   # must always have 1 or more features

    try:
        list(iter(_poly_combos))
        assert all(map(isinstance, _poly_combos, (tuple for i in _poly_combos)))
    except Exception as e:
        raise AssertionError

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

    assert isinstance(_job_size, numbers.Integral)
    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    args = {'_rtol': _rtol, '_atol': _atol, '_equal_nan': _equal_nan}


    # convert combos to np for slicing out poly combos
    _poly_combos = np.array(list(map(tuple, _poly_combos)), dtype=object)
    # look for duplicates in X v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v
    # there can be more than one hit for duplicates in X

    _X_dupls = []
    if _min_degree == 1:

        _asymmetric_combinations = []
        for i in range(math.ceil(_X.shape[1]/_job_size)):   # _X chunks
            for j in range(i, math.ceil(len(_poly_combos)/_job_size)):   # _combos chunks
                _asymmetric_combinations.append((i, j))

        if _X.shape[1] < 2 * _job_size:
            for i, j in _asymmetric_combinations:  # i is X, j is combos
                _X1_idxs = tuple((k,) for k in range(i * _job_size, min((i + 1) * _job_size, _X.shape[1])))
                _X2_idxs = tuple(map(tuple, _poly_combos[list(range(j * _job_size, min((j + 1) * _job_size, len(_poly_combos))))]))
                _X_dupls.append(
                    _parallel_chunk_comparer(
                        _chunk1=_columns_getter(_X, _X1_idxs),
                        _chunk1_X_indices=_X1_idxs,
                        _chunk2=_columns_getter(_X, _X2_idxs),
                        _chunk2_X_indices=_X2_idxs,
                        **args
                    )
                )
            del _X1_idxs, _X2_idxs
        else:
            with joblib.parallel_config(prefer='processes', n_jobs=_n_jobs):
                _X_dupls = joblib.Parallel(return_as='list')(
                    joblib.delayed(_parallel_chunk_comparer)(
                        _chunk1=_columns_getter(
                            _X,
                            tuple((k,) for k in range(i*_job_size,  min((i+1)*_job_size, _X.shape[1])))
                        ),
                        _chunk1_X_indices=tuple((k,) for k in range(i*_job_size,  min((i+1)*_job_size, _X.shape[1]))),
                        _chunk2=_columns_getter(
                            _X,
                            tuple(map(tuple, _poly_combos[list(range(j*_job_size,  min((j+1)*_job_size, len(_poly_combos))))]))
                        ),
                        _chunk2_X_indices=tuple(
                            tuple(map(tuple, _poly_combos[list(range(j*_job_size,  min((j+1)*_job_size, len(_poly_combos))))]))
                        ),
                        **args
                    ) for i, j in _asymmetric_combinations   # i is X, j is combos
                )

        _X_dupls = list(itertools.chain(*_X_dupls))

        del _asymmetric_combinations

    else:
        _X_dupls = []

    # END look for duplicates in X v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v

    # look for duplicates in POLY v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^

    _symmetric_combinations = list(itertools.combinations_with_replacement(
        range(math.ceil(len(_poly_combos)/_job_size)),
        2
    ))

    _poly_dupls = []
    if len(_poly_combos) < 2 * _job_size:
        for i, j in _symmetric_combinations:
            _X1_idxs = tuple(map(tuple, _poly_combos[list(range(i * _job_size, min((i+1) * _job_size, len(_poly_combos))))]))
            _X2_idxs = tuple(map(tuple, _poly_combos[list(range(j * _job_size, min((j+1) * _job_size, len(_poly_combos))))]))
            _poly_dupls.append(_parallel_chunk_comparer(
                _chunk1=_columns_getter(_X, _X1_idxs),
                _chunk1_X_indices=_X1_idxs,
                _chunk2=_columns_getter(_X, _X2_idxs),
                _chunk2_X_indices=_X2_idxs,
                **args
                )
            )
        del _X1_idxs, _X2_idxs
    else:
        with joblib.parallel_config(prefer='processes', n_jobs=_n_jobs):
            _poly_dupls = joblib.Parallel(return_as='list')(
                joblib.delayed(_parallel_chunk_comparer)(
                    _chunk1=_columns_getter(
                        _X,
                        tuple(map(tuple, _poly_combos[list(range(i * _job_size, min((i+1) * _job_size, len(_poly_combos))))]))
                    ),
                    _chunk1_X_indices=tuple(map(tuple, _poly_combos[list(range(i * _job_size, min((i+1) * _job_size, len(_poly_combos))))])),
                    _chunk2=_columns_getter(
                        _X,
                        tuple(map(tuple, _poly_combos[list(range(j * _job_size, min((j+1) * _job_size, len(_poly_combos))))]))
                    ),
                    _chunk2_X_indices=tuple(map(tuple, _poly_combos[list(range(j * _job_size, min((j+1) * _job_size, len(_poly_combos))))])),
                    **args
                ) for i, j in _symmetric_combinations
            )

    del _symmetric_combinations

    _poly_dupls = list(itertools.chain(*_poly_dupls))

    # END look for duplicates in POLY v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^


    _all_dupls = _X_dupls + _poly_dupls


    return _all_dupls




