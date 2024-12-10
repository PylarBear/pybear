# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union

import numbers

import scipy.sparse as ss
from joblib import Parallel, delayed, wrap_non_picklable_objects

from .._partial_fit._columns_getter import _columns_getter








def _build_poly(
    X: ss.csc_array,
    _active_combos: tuple[tuple[int, ...], ...],
    _n_jobs: Union[numbers.Integral, None]
) -> ss.csc_array:

    """
    Pizza. Build the polynomial expansion for X as a scipy sparse csc
    array. Index tuples in :param: _combos that are not in :param:
    dropped_poly_duplicates_ are omitted from the expansion.


    Parameters
    ----------
    X:
        {scipy sparse csc_array} of shape (n_samples,
        n_features) - The data to be expanded.
    __active_combos:
        tuple[tuple[int, ...], ...] - the index tuple combinations to be
        kept in the polynomial expansion.
    _n_jobs:
        Union[numbers.Integral, None] - the number of parallel jobs to
        use when building the polynomial expansion.


    Return
    ------
    -
        POLY: scipy sparse csc array of shape (n_samples, n_kept_polynomial_features) -
        The polynomial expansion.

    """


    # validation - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    assert isinstance(X, ss.csc_array)
    assert isinstance(_active_combos, tuple)
    for _tuple in _active_combos:
        assert isinstance(_tuple, tuple)
        assert all(map(isinstance, _tuple, (int for _ in _tuple)))
    assert isinstance(_n_jobs, (numbers.Integral, type(None)))
    assert _n_jobs >= -1 and _n_jobs != 0
    # END validation - - - - - - - - - - - - - - - - - - - - - - - - - - - -



    @wrap_non_picklable_objects
    def _poly_stacker(_columns):
        return ss.csc_array(_columns.prod(1))


    # pizza, do a benchmark on this, is it faster to just do a for loop
    # with all this serialization?
    joblib_kwargs = {'prefer': 'processes', 'return_as': 'list', 'n_jobs': _n_jobs}
    out = Parallel(**joblib_kwargs)(
        delayed(
            _poly_stacker(_columns_getter(X, combo))
        ) for combo in _active_combos
    )


    POLY = ss.hstack(out)


    return POLY







