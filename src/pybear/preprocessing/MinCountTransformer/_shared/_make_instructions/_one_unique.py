# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing_extensions import Union, Literal
from ..._type_aliases import DataType




def _one_unique(
    _instr_list: list,
    _threshold: int,
    _nan_key: Union[float, str, Literal[False]],
    _nan_ct: Union[int, Literal[False]],
    _COLUMN_UNQ_CT_DICT: dict[DataType, int],
    ) -> list[Union[str, DataType]]:

    """
    Make delete instructions for a column with one unique non-nan value.

    WHEN 1 item (NOT INCLUDING nan):
    - if not ign nan and has nans
        If not ignoring nan and ct_nan < thresh, delete the nan rows but
        do not delete the other rows (would delete all rows.) If count of
        non-nan value is ever below thresh, never delete rows, just delete
        column.
    - if ignoring nan or there are no nans
        Do not delete impacted rows no matter what the dtype was or what
        kwargs were given, would delete all rows.

    Parameters
    ----------
    _instr_list: list, should be empty
    _threshold: int
    _nan_key: Union[float, str, Literal[False]]
    _nan_ct: Union[int, Literal[False]]
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

    if len(_COLUMN_UNQ_CT_DICT) > 1:
        raise ValueError(f"len(_UNQ_CTS_DICT) > 1")

    if (_nan_ct is False) + (_nan_key is False) not in [0, 2]:
        raise ValueError(f"_nan_key is {_nan_key} and _nan_ct is {_nan_ct}")


    if _nan_ct:
        # if not ign nan and has nans (it must be the only value in the column!)
        # 24_06_11_14_37_00, if nan ct < thresh, do not delete the rows
        # if _nan_ct < _threshold:
        #     _instr_list.append(_nan_key)

        _instr_list.append('DELETE COLUMN')


    elif not _nan_ct:
        # if ignoring nan or there are no nans
        _instr_list.append('DELETE COLUMN')


    return _instr_list















