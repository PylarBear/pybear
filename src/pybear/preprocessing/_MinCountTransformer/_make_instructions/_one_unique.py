# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#s



from typing_extensions import Union, Literal
from .._type_aliases import DataType



def _one_unique(
    _threshold: int,
    _nan_key: Union[float, str, Literal[False]],
    _nan_ct: Union[int, Literal[False]],
    _COLUMN_UNQ_CT_DICT: dict[DataType, int],
) -> list[Union[Literal['DELETE COLUMN'], DataType]]:

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
    _threshold:
        int - the threshold value for the selected column
    _nan_key:
        Union[float, str, Literal[False]] - the nan value found in the
        data. as of 25_01 all nan-likes are converted to numpy.nan by
        _columns_getter.
    _nan_ct:
        Union[int, Literal[False]] - the number of nans found in this
        column.
    _COLUMN_UNQ_CT_DICT:
        dict[DataType, int] - the value from _total_cts_by_column for
        this column which is a dictionary that holds the uniques and
        their frequencies. cannot be empty.


    Return
    ------
    -
        _instr_list: list[Union[DataType, Literal['DELETE COLUMN']] -
        the row and column operation instructions for this column.

    """

    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    if 'nan' in list(map(str.lower, map(str, _COLUMN_UNQ_CT_DICT.keys()))):
        raise ValueError(f"nan-like is in _UNQ_CTS_DICT and should not be")

    if len(_COLUMN_UNQ_CT_DICT) > 1:
        raise ValueError(f"len(_UNQ_CTS_DICT) > 1")

    if (_nan_ct is False) + (_nan_key is False) not in [0, 2]:
        raise ValueError(f"_nan_key is {_nan_key} and _nan_ct is {_nan_ct}")

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    if _nan_ct:
        # if not ign nan and has nans, it must be the only value in the column
        _instr_list = ['DELETE COLUMN']
    elif not _nan_ct:
        # if ignoring nan or there are no nans
        _instr_list = ['DELETE COLUMN']


    return _instr_list















