# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.SlimPolyFeatures._type_aliases import DataType
import numpy.typing as npt
from typing_extensions import Union

import numpy as np
import pandas as pd

from pybear.utilities._nan_masking import nan_mask



def _columns_getter(
    _DATA: DataType,
    _col_idxs: Union[int, tuple[int, ...]]
) -> npt.NDArray[any]:

    """
    Handles the mechanics of extracting one or more columns from the
    various allowed data container types. Return extracted columns as a
    numpy array in row-major order. Scipy sparse formats must be
    indexable.


    Parameters
    ----------
    _DATA:
        Union[NDArray, pd.Dataframe, scipy.sparse] - The data to extract
        columns from.
    _col_idxs:
        Union[int, tuple[int, ...]] - the column index / indices to
        extract from the data.


    Return
    ------
    -
        _columns: NDArray - The columns from the data corresponding to
        the given indices in row-major order.

    """

    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    assert isinstance(_DATA, (np.ndarray, pd.core.frame.DataFrame)) or \
        hasattr(_DATA, 'toarray')

    assert isinstance(_col_idxs, (int, tuple))
    if isinstance(_col_idxs, int):
        _col_idxs = (_col_idxs,)   # <==== int _col_idx converted to tuple
    assert len(_col_idxs), f"'_col_idxs' cannot be empty"
    for _idx in _col_idxs:
        assert isinstance(_idx, int)
        assert _idx in range(_DATA.shape[1]), f"col idx out of range"
    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    _col_idxs = sorted(list(_col_idxs))

    if isinstance(_DATA, np.ndarray):
        _columns = _DATA[:, _col_idxs]
    elif isinstance(_DATA, pd.core.frame.DataFrame):
        # additional steps are taken at the bottom of this module if the
        # dataframe has funky nan-likes, causing _columns to leave this
        # step as dtype object
        _columns = _DATA.iloc[:, _col_idxs].to_numpy()
    elif hasattr(_DATA, 'toarray'):
        # both _parallel_constant_finder() and _build_poly() need vectors
        # extracted from ss to be full, not the stacked version
        # (ss.indices & ss.data hstacked). With all the various
        # applications that use _columns_getter, and all the various
        # forms that could be needed at those endpoints, it is simplest
        # just to standardize all to receive dense np, at the cost of
        # slightly higher memory swell than may otherwise be necessary.
        # Extract the columns from scipy sparse as dense ndarray.
        _columns = _DATA[:, _col_idxs].toarray()
    else:
        raise TypeError(f"invalid data type '{type(_DATA)}'")



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

    _columns = _columns.astype(np.float64)


    return _columns








