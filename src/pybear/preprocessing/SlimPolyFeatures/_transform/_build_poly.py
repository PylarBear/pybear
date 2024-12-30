# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
from .._type_aliases import DataType

import numbers
import numpy as np
import pandas as pd
import scipy.sparse as ss
from joblib import Parallel, delayed, wrap_non_picklable_objects

from .._partial_fit._columns_getter import _columns_getter




def _build_poly(
    X: DataType,
    _active_combos: tuple[tuple[int, ...], ...],
    _n_jobs: Union[numbers.Integral, None]
) -> ss.csc_array:

    """
    Build the polynomial expansion for X as a scipy sparse csc array
    using X and _active_combos. X is as received into SPF :method:
    transform, that is, it can be np.ndarray, pd.DataFrame, or any
    scipy sparse. _active_combos is all combinations from the original
    combinations that are not in dropped_poly_duplicates_ and
    poly_constants_.


    Parameters
    ----------
    X:
        {np.ndarray, pd.DataFrame, scipy sparse} of shape (n_samples,
        n_features) - The data to undergo polynomial expansion.
    _active_combos:
        tuple[tuple[int, ...], ...] - the index tuple combinations to be
        kept in the final polynomial expansion.
    _n_jobs:
        Union[numbers.Integral, None] - the number of parallel jobs to
        use when building the polynomial expansion.


    Return
    ------
    -
        POLY: scipy sparse csc array of shape (n_samples,
        n_kept_polynomial_features) - The polynomial expansion component
        of the final output.

    """


    # validation - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    assert isinstance(X, (np.ndarray, pd.core.frame.DataFrame)) or \
           hasattr(X, 'toarray'), f"{type(X)=}"

    assert isinstance(_active_combos, tuple)
    for _tuple in _active_combos:
        assert isinstance(_tuple, tuple)
        assert all(map(isinstance, _tuple, (int for _ in _tuple)))

    assert isinstance(_n_jobs, (numbers.Integral, type(None)))
    assert not isinstance(_n_jobs, bool)
    if _n_jobs is not None:
        assert _n_jobs >= -1 and _n_jobs != 0
    # END validation - - - - - - - - - - - - - - - - - - - - - - - - - - - -


    if not len(_active_combos):
        POLY = ss.csc_array(np.empty((X.shape[0], 0), dtype=np.float64))
        return POLY


    @wrap_non_picklable_objects
    def _poly_stacker(_columns):
        return ss.csc_array(_columns.prod(1).reshape((-1,1)))


    # pizza, do a benchmark on this, is it faster to just do a for loop
    # with all this serialization?
    joblib_kwargs = {
        'prefer': 'processes', 'return_as': 'list', 'n_jobs': _n_jobs
    }
    POLY = Parallel(**joblib_kwargs)(delayed(_poly_stacker)(
        _columns_getter(X, combo)) for combo in _active_combos
    )

    POLY = ss.hstack(POLY).astype(np.float64)


    assert POLY.shape == (X.shape[0], len(_active_combos))


    return POLY




























