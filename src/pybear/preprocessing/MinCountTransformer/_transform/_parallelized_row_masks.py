# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy as np
import joblib
from typing_extensions import Union
from .._type_aliases import DataType
from ....utilities._nan_masking import nan_mask



@joblib.wrap_non_picklable_objects
def _parallelized_row_masks(
    _X_COLUMN: np.ndarray[DataType],
    _COLUMN_UNQ_CT_DICT: dict[DataType, int],
    _instr: list[Union[str, DataType]],
    _reject_unseen_values: bool,
    _col_idx: int
    ) -> np.ndarray[np.uint8]:


    """
    Create mask indicating row positions to delete for one column of data
    from X. Using the instructions provided in _instr, locate the positions
    of each unique to be deleted and store the locations as a vector that
    is the same size as the column.

    Simultaneously, if rejecting unseen values, compare the uniques found
    in _COLUMN_UNQ_CT_DICT against the uniques found in the column.
    Raise exception if rejecting unseen values and there are new uniques.

    Parameters
    ----------
    _X_COLUMN: np.ndarray[DataType] - A column of data from X
    _COLUMN_UNQ_CT_DICT: dict[DataType, int] - The entry for this column
        in _total_counts_by_column
    _instr: list[Union[str, DataType]] - The entry for the this column
        in _delete_instr
    _reject_unseen_values: bool - If False, do not even look to see if
        there are unknown uniques in the column. If True, compare uniques
        in the column against uniques in _COLUMN_UNQ_CT_DICT and raise
        exception if there is a value not previously seen.
    _col_idx: int - The index that this column occupied in X. For error
        reporting purposes only.

    Return
    ------
    -
        COLUMN_ROW_MASK: np.ndarray[int]: Mask of rows to delete based on
            the instructions in _delete_instr for that column

    """

    assert isinstance(_X_COLUMN, np.ndarray)
    assert _X_COLUMN.shape[1] == 1
    assert isinstance(_COLUMN_UNQ_CT_DICT, dict)
    assert all(map(
        isinstance,
        _COLUMN_UNQ_CT_DICT.values(),
        (int for _ in _COLUMN_UNQ_CT_DICT)
    ))
    assert isinstance(_instr, list)
    assert isinstance(_reject_unseen_values, bool)
    assert isinstance(_col_idx, int)


    COLUMN_ROW_MASK = np.zeros(_X_COLUMN.shape[0], dtype=np.uint8)

    RUV_MASK = np.zeros(_X_COLUMN.shape[0], dtype=np.uint8)

    _nan_ctr = 0
    for unq in _COLUMN_UNQ_CT_DICT:

        MASK_ON_X_COLUMN_UNQ = np.zeros(_X_COLUMN.shape[0], dtype=np.uint8)

        if unq in _instr or str(unq) in map(str, _instr) or _reject_unseen_values:

            if str(unq).lower() == 'nan':
                _nan_ctr += 1
                if _nan_ctr > 1:
                    raise ValueError(f"more than one nan-like in UNQ_CT_DICT")
                MASK_ON_X_COLUMN_UNQ += nan_mask(_X_COLUMN.ravel())
            else:
                MASK_ON_X_COLUMN_UNQ += (_X_COLUMN.ravel() == unq)

        if _reject_unseen_values:
            RUV_MASK += MASK_ON_X_COLUMN_UNQ.astype(np.uint8)

        if unq in _instr or str(unq) in map(str, _instr):
            COLUMN_ROW_MASK += MASK_ON_X_COLUMN_UNQ.astype(np.uint8)

    del _nan_ctr, MASK_ON_X_COLUMN_UNQ

    # 24_10_20, python sum is not working correctly on RUV_MASK when has a
    # np dtype, need to use np sum to get the correct result. Or, could
    # convert RUV_MASK with .astype(int), and py sum works correctly.
    # Could go either way with this fix.

    if _reject_unseen_values and np.sum(RUV_MASK) != _X_COLUMN.shape[0]:

        # build things to display info about unseen values ** * ** * **
        _X = _X_COLUMN[np.logical_not(RUV_MASK), :]
        orig_dtype = _X_COLUMN.dtype

        try:
            _UNSEEN_UNQS = np.unique(_X.astype(np.float64)).astype(orig_dtype)
        except:
            _UNSEEN_UNQS = np.unique(_X.astype(str)).astype(orig_dtype)

        del orig_dtype, _X
        # END build things to display info about unseen values ** * ** * **

        if len(_UNSEEN_UNQS) > 10:
            _UNSEEN_UNQS = f"{_UNSEEN_UNQS[:10]} + others"

        raise ValueError(f"Transform data has values not seen "
            f"during fit --- "
            f"\ncolumn index = {_col_idx}"
            f"\nunseen values = {_UNSEEN_UNQS}")

    del RUV_MASK


    return COLUMN_ROW_MASK

























