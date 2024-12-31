# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import DataType

import numpy as np
import pandas as pd
import scipy.sparse as ss

from .._partial_fit._columns_getter import _columns_getter




def _build_poly(
    X: DataType,
    _active_combos: tuple[tuple[int, ...], ...]
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
    # END validation - - - - - - - - - - - - - - - - - - - - - - - - - - - -


    if not len(_active_combos):
        POLY = ss.csc_array(np.empty((X.shape[0], 0), dtype=np.float64))
        return POLY


    POLY = ss.csc_array(np.empty((X.shape[0], 0))).astype(np.float64)
    for combo in _active_combos:
        _poly_feature = _columns_getter(X, combo).prod(1).reshape((-1, 1))
        POLY = ss.hstack((POLY, ss.csc_array(_poly_feature)))

    assert POLY.shape == (X.shape[0], len(_active_combos))


    # LINUX TIME TRIALS INDICATE A REGULAR FOR LOOP IS ABOUT HALF THE
    # TIME OF JOBLIB
    # @wrap_non_picklable_objects
    # def _poly_stacker(_columns):
    #     return ss.csc_array(_columns.prod(1).reshape((-1,1)))
    # joblib_kwargs = {
    #     'prefer': 'processes', 'return_as': 'list', 'n_jobs': _n_jobs
    # }
    # POLY = Parallel(**joblib_kwargs)(delayed(_poly_stacker)(
    #     _columns_getter(X, combo)) for combo in _active_combos
    # )
    #
    # POLY = ss.hstack(POLY).astype(np.float64)
    # assert POLY.shape == (X.shape[0], len(_active_combos))


    return POLY




























