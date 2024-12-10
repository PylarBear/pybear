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
import scipy.sparse as ss

from pybear.utilities._nan_masking import nan_mask



def _columns_getter(
    _DATA: DataType,
    _col_idxs: Union[int, tuple[int, ...]]
) -> npt.NDArray[any]:

    """
    Handles the mechanics of extracting one or more columns from the
    various data container types. Return extracted columns as a numpy
    array.


    Parameters
    ----------
    _DATA:
        DataType - The data to extract columns from.
    _col_idxs:
        int - the first column index in the comparison pair.


    Return
    ------
    -
        _columns: NDArray[any] - The columns corresponding to the given
        indices.

    """


    assert isinstance(_col_idxs, (int, tuple))
    if isinstance(_col_idxs, int):
        _col_idxs = (_col_idxs,)
    assert len(_col_idxs), f"'_col_idxs' cannot be empty"
    for _idx in _col_idxs:
        assert isinstance(_idx, int)
        assert _idx in range(_DATA.shape[1]), f"col idx out of range"

    _col_idxs = sorted(list(_col_idxs))



    if isinstance(_DATA, np.ndarray):
        _columns = _DATA[:, _col_idxs]
    elif isinstance(_DATA, pd.core.frame.DataFrame):
        _columns = _DATA.iloc[:, _col_idxs].to_numpy()
    elif hasattr(_DATA, 'toarray'):
        """
        # instead of expanding the column to dense np, get the indices
        # and values out of sparse column using the 'indices' and 'data'
        # attributes, hstack them, and send that off for equality test

        if isinstance(_DATA, (ss.dia_matrix, ss.dia_array)):
            # for some unknown reason, when tocsc() is applied to dia,
            # the indices come out sorted descending. but if u do tocsr()
            # before tocsc(), then the indices come out sorted ascending.
            # this really only serves to standardize the dia output with
            # the other scipy sparse formats, and makes testing easier.
            c1 = _DATA.tocsr().tocsc()[:, _col_idxs]
        else:
            # Extract the data and indices of the column
            c1 = _DATA.tocsc()[:, _col_idxs]
        # reshaping standardizes the output with np and pd
        # pizza, what kind of crack were u on?
        _columns = np.hstack((c1.indices, c1.data)).reshape((-1, 1))
        del c1
        
        
        pizza 24_12_09_19_44_00, looks like both _parallel_constant_finder() 
        and _build_poly need vectors extracted from ss to be full, not the 
        stacked version. if this condition is still true at the finish, 
        then delete all this jive.         
        """

        # old code that converts a ss column to np array
        _columns = _DATA.copy().tocsc()[:, _col_idxs].toarray()
    else:
        raise TypeError(f"invalid data type '{type(_DATA)}'")


    # pizza verify this!
    # this assignment must stay here. there was a nan recognition problem
    # that wasnt happening in offline tests of entire data objects
    # holding the gamut of nan-likes but was happening with similar data
    # objects passing thru the SlimPoly machinery. Dont know the reason why,
    # maybe because the columns get parted out, or because they get sent
    # thru the joblib machinery? using nan_mask here and reassigning all
    # nans identified here as np.nan resolved the issue.
    # np.nan assignment excepts on dtype int array, so ask for forgiveness
    try:
        _columns[nan_mask(_columns)] = np.nan
    except:
        pass

    return _columns








