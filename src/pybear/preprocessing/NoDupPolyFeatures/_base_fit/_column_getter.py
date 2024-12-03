# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.NoDupPolyFeatures._type_aliases import DataType
import numpy.typing as npt
from typing_extensions import Union

import numpy as np
import pandas as pd

from pybear.utilities._nan_masking import nan_mask



def _column_getter(
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
        _columns: NDArray[any] - The columns corresponding to the given indices.

    """


    assert isinstance(_col_idxs, (int, tuple))
    if isinstance(_col_idxs, int):
        _col_idxs = (_col_idxs,)
    assert len(_col_idxs), f"'_col_idxs' cannot be empty"
    for _idx in _col_idxs:
        assert isinstance(_idx, int)
        assert _idx in range(_DATA.shape[1]), f"col idx out of range"





    if isinstance(_DATA, np.ndarray):
        _columns = _DATA[:, list(_col_idxs)]
    elif isinstance(_DATA, pd.core.frame.DataFrame):
        _columns = _DATA.iloc[:, list(_col_idxs)].to_numpy()
    elif hasattr(_DATA, 'toarray'):
        _columns = _DATA.copy().tocsc()[:, list(_col_idxs)].toarray()
    else:
        raise TypeError(f"invalid data type '{type(_DATA)}'")


    # pizza verify this!
    # this assignment must stay here. there was a nan recognition problem
    # that wasnt happening in offline tests of entire data objects
    # holding the gamut of nan-likes but was happening with similar data
    # objects passing thru the NoDup machinery. Dont know the reason why,
    # maybe because the columns get parted out, or because they get sent
    # thru the joblib machinery? using nan_mask here and reassigning all
    # nans identified here as np.nan resolved the issue.
    # np.nan assignment excepts on dtype int array, so ask for forgiveness
    try:
        _columns[nan_mask(_columns)] = np.nan
    except:
        pass

    return _columns








