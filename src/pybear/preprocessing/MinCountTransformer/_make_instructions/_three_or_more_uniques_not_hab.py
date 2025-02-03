# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union, Literal
from .._type_aliases import DataType

import numpy as np



def _three_or_more_uniques_not_hab(
    _threshold: int,
    _nan_key: Union[float, str, Literal[False]],
    _nan_ct: Union[int,  Literal[False]],
    _COLUMN_UNQ_CT_DICT: dict[DataType, int],
) -> list[Union[Literal['DELETE ALL', 'DELETE COLUMN'], DataType]]:

    """
    Make delete instructions for a column with three or more unique
    non-nan values.

    if not _handle_as_bool, _delete_axis_0 doesnt matter, always delete

    WHEN 3 items (NOT INCLUDING nan):
    if no nans or ignoring
        look over all counts
        any ct below threshold mark to delete rows
        if one or less non-nan thing left in column, DELETE COLUMN
    if not ignoring nans
        look over all counts
        any ct below threshold mark to delete rows
        if nan ct below threshold, mark to delete rows
        if only 1 or 0 non-nans left in column, DELETE COLUMN


    Parameters
    ----------
    _threshold:
        int - the minimum frequency threshold for this column
    _nan_key:
        Union[float, str, Literal[False]] - the nan value found in the
        column in its original dtype. as of 25_01 _column_getter is
        converting all nan-like values to numpy.nan.
    _nan_ct:
        Union[int,  Literal[False]] - the number of nan-like value found
        in the column.
    _COLUMN_UNQ_CT_DICT:
        dict[DataType, int] - the value from _total_cts_by_column for
        this column which is a dictionary that holds the uniques and
        their frequencies. cannot be empty.


    Return
    ------
    -
        _instr_list:
            list[Union[Literal['DELETE ALL', 'DELETE COLUMN', DataType]] -
            the row and columns operations to be performed for this
            column.


    """

    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    if 'nan' in list(map(str.lower, map(str, _COLUMN_UNQ_CT_DICT.keys()))):
        raise ValueError(f"nan-like is in _UNQ_CTS_DICT and should not be")

    if not len(_COLUMN_UNQ_CT_DICT) >= 3:
        raise ValueError(f"len(_UNQ_CTS_DICT) not >= 3")

    if (_nan_ct is False) + (_nan_key is False) not in [0, 2]:
        raise ValueError(f"_nan_key is {_nan_key} and _nan_ct is {_nan_ct}")

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # _delete_axis_0 DOES NOT APPLY, MUST DELETE ALONG AXIS 0
    # IF ONLY 1 UNQ LEFT, DELETE COLUMN,
    # IF ALL UNQS DELETED THIS SHOULD PUT ALL False IN
    # ROW MASK AND CAUSE EXCEPT DURING transform()
    _instr_list = []
    CTS = np.fromiter(_COLUMN_UNQ_CT_DICT.values(), dtype=np.uint32)
    if np.sum((CTS < _threshold)) == len(CTS):
        _instr_list.append('DELETE ALL')
        del CTS
    else:
        # do this the long way, not by slicing numpy vectors which will
        # turn everything to stuff like np.str_('a'), to preserve
        # the original format of the unqs.
        for unq, ct in _COLUMN_UNQ_CT_DICT.items():
            if ct < _threshold:
                _instr_list.append(unq)

    # must get len(_instr_list) before (potentially) putting nan_key in it
    _delete_column = False
    if 'DELETE ALL' in _instr_list \
            or len(_instr_list) >= len(_COLUMN_UNQ_CT_DICT) - 1:
        _delete_column = True

    if 'DELETE ALL' not in _instr_list and _nan_ct and _nan_ct < _threshold:
        _instr_list.append(_nan_key)

    if _delete_column:
        _instr_list.append('DELETE COLUMN')

    del _delete_column


    return _instr_list











