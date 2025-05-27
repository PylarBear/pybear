# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
import numpy.typing as npt
from pybear.preprocessing._SlimPolyFeatures._type_aliases import \
    InternalDataContainer

import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as ss

from pybear.utilities._nan_masking import nan_mask



def _columns_getter(
    _X: InternalDataContainer,
    _col_idxs: Union[int, tuple[int, ...]]
) -> npt.NDArray[np.float64]:

    """
    Handles the mechanics of extracting one or more columns from the
    various allowed data container types. Data passed as scipy sparse
    formats must be indexable. Therefore, coo matrix/array, dia
    matrix/array, and bsr matrix/array are prohibited. Return extracted
    column(s) as a numpy array in row-major order. In the case of scipy
    sparse, the columns are converted to dense.


    Parameters
    ----------
    _X:
        array-like - The data to extract columns from. _X must be
        indexable, which excludes scipy coo, dia, and bsr. This module
        expects _X to be in a valid state when passed, and will not
        condition it.
    _col_idxs:
        Union[int, tuple[int, ...]] - the column index / indices to
        extract from _X.


    Return
    ------
    -
        _columns: NDArray[np.float64] - The columns from _X corresponding
        to the given indices in row-major order.

    """

    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    assert isinstance(_X, (np.ndarray, pd.core.frame.DataFrame, pl.DataFrame)) or \
        hasattr(_X, 'toarray')

    assert not isinstance(_X,
        (ss.coo_matrix, ss.coo_array, ss.dia_matrix,
         ss.dia_array, ss.bsr_matrix, ss.bsr_array)
    )

    assert isinstance(_col_idxs, (int, tuple))
    if isinstance(_col_idxs, int):
        _col_idxs = (_col_idxs,)   # <==== int _col_idx converted to tuple
    assert len(_col_idxs), f"'_col_idxs' cannot be empty"
    for _idx in _col_idxs:
        assert isinstance(_idx, int)
        assert _idx in range(_X.shape[1]), f"col idx out of range"
    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    _col_idxs = sorted(list(_col_idxs))

    if isinstance(_X, np.ndarray):
        _columns = _X[:, _col_idxs]
    elif isinstance(_X, pd.core.frame.DataFrame):
        # additional steps are taken at the bottom of this module if the
        # dataframe has funky nan-likes, causing _columns to leave this
        # step as dtype object
        _columns = _X.iloc[:, _col_idxs].to_numpy()
    elif isinstance(_X, pl.DataFrame):
        # pizza 25_05_26
        # when pulling the same column 2+ times, polars cannot made df
        # polars.exceptions.DuplicateError: could not create a new DataFrame:
        # column with name 'd61193cc' has more than one occurrence
        # need a workaround that doesnt copy the full X.
        # pull the unique columns, convert to np, then create the og stack
        _unq_idxs = np.unique(_col_idxs)
        # need to map idxs in X to future idxs in the uniques slice
        _lookup_dict = {}
        for _new_idx, _old_idx in enumerate(_unq_idxs):
            _lookup_dict[_old_idx] = _new_idx
        _columns = _X[:, _unq_idxs].to_numpy()
        _new_idxs = [int(_lookup_dict[_old_idx]) for _old_idx in _col_idxs]
        _columns = _columns[:, _new_idxs]
    elif hasattr(_X, 'toarray'):
        # both _is_constant() and _build_poly() need vectors
        # extracted from ss to be full, not the stacked version
        # (ss.indices & ss.data hstacked). With all the various
        # applications that use _columns_getter, and all the various
        # forms that could be needed at those endpoints, it is simplest
        # just to standardize all to receive dense np, at the cost of
        # slightly higher memory swell than may otherwise be necessary.
        # Extract the columns from scipy sparse as dense ndarray.
        _columns = _X[:, _col_idxs].toarray()
    else:
        raise TypeError(f"invalid data type '{type(_X)}'")


    # this assignment must stay here. there was a nan recognition problem
    # that wasnt happening in offline tests of entire data objects
    # holding the gamut of nan-likes but was happening with similar data
    # objects passing thru the SlimPoly machinery. Dont know the reason
    # why, maybe because the columns get parted out, or because they get
    # sent thru the joblib machinery? using nan_mask here and reassigning
    # all nans identified here as np.nan resolved the issue.
    # np.nan assignment excepts on dtype int array, so ask for forgiveness
    try:
        _columns[nan_mask(_columns)] = np.nan
    except:
        pass

    # if pandas had funky nan-likes (or if X was np, ss, or pd, and
    # dtype was passed as object), then the dtype of _columns is
    # guaranteed to be object and ss cant take it. need to convert the
    # dtype to a numeric one. ultimately, for several reasons, the
    # executive decision was made to always build POLY as float64. if
    # there are nans in this, then it must be np.float64 anyway.

    _columns = np.ascontiguousarray(_columns).astype(np.float64)

    return _columns






