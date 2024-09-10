# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from typing_extensions import Union, Literal
from ..._type_aliases import DataType

import numpy as np



def _three_or_more_uniques_not_hab(
        _instr_list: list,
        _threshold: int,
        _nan_key: Union[float, str, Literal[False]],
        _nan_ct: Union[int,  Literal[False]],
        _COLUMN_UNQ_CT_DICT: dict[DataType, int],
    ) -> list[Union[str, DataType]]:

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
    _instr_list: list, should be empty
    _threshold: int
    _nan_key: Union[float, str, Literal[False]]
    _nan_ct: Union[int,  Literal[False]]
    _COLUMN_UNQ_CT_DICT: dict[DataType, int], cannot be empty

    Return
    ------
    -
        _instr_list: list[Union[str, DataType]]


    """

    if not len(_instr_list) == 0:
        raise ValueError(f"'_instr_list' must be empty")

    if 'nan' in list(map(str.lower, map(str, _COLUMN_UNQ_CT_DICT.keys()))):
        raise ValueError(f"nan-like is in _UNQ_CTS_DICT and should not be")

    if not len(_COLUMN_UNQ_CT_DICT) >= 3:
        raise ValueError(f"len(_UNQ_CTS_DICT) not >= 3")

    if (_nan_ct is False) + (_nan_key is False) not in [0, 2]:
        raise ValueError(f"_nan_key is {_nan_key} and _nan_ct is {_nan_ct}")


    if not _nan_ct:  # EITHER IGNORING NANS OR NONE IN FEATURE

        # _delete_axis_0 NO LONGER APPLIES, MUST DELETE ALONG AXIS 0
        # IF ONLY 1 UNQ LEFT, DELETE COLUMN,
        # IF ALL UNQS DELETED THIS SHOULD PUT ALL False IN
        # ROW MASK AND CAUSE EXCEPT DURING transform()
        for unq, ct in _COLUMN_UNQ_CT_DICT.items():
            if ct < _threshold:
                _instr_list.append(unq)

        if len(_instr_list) >= len(_COLUMN_UNQ_CT_DICT) - 1:
            _instr_list.append('DELETE COLUMN')

        del unq, ct


    else:  # HAS NANS AND NOT IGNORING

        # _delete_axis_0 NO LONGER APPLIES,
        # MUST DELETE ALONG AXIS 0
        # IF ONLY 1 UNQ LEFT, DELETE COLUMN,
        # IF ALL UNQS DELETED THIS SHOULD PUT ALL False IN
        # ROW MASK AND CAUSE EXCEPT DURING transform()
        for unq, ct in _COLUMN_UNQ_CT_DICT.items():
            if ct < _threshold:
                _instr_list.append(unq)

        _delete_column = False
        if len(_instr_list) >= len(_COLUMN_UNQ_CT_DICT) - 1:
            _delete_column = True

        if _nan_ct < _threshold:
            _instr_list.append(_nan_key)

        if _delete_column:
            _instr_list.append('DELETE COLUMN')

        del unq, ct, _delete_column



    return _instr_list











