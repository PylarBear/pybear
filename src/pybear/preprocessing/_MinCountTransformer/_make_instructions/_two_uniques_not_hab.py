# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
from typing import Literal
from .._type_aliases import DataType



def _two_uniques_not_hab(
    _threshold: int,
    _nan_key: Union[float, str, Literal[False]],
    _nan_ct: Union[int, Literal[False]],
    _COLUMN_UNQ_CT_DICT: dict[DataType, int]
) -> list[Union[Literal['DELETE COLUMN'], DataType]]:


    """

    Make delete instructions for a column with two unique non-nan values.

    WHEN 2 items (NOT INCLUDING nan):
    *** BINARY INT COLUMNS ARE HANDLED DIFFERENTLY THAN OTHER DTYPES ***
    Most importantly, if a binary integer has a value below threshold,
    the DEFAULT behavior is to not delete the respective rows (whereas
    any other dtype will delete the rows), essentially causing bin int
    columns with insufficient count to just be deleted.

    - if ignoring nan or no nans
        -- look at cts for the 2 unqs, if any ct < thresh, mark rows
            for deletion
        -- if any below thresh, DELETE COLUMN
    - if not ign nan and has nans
        -- put nan & ct back in dict, treat it like any other value
        -- look at cts for the 3 unqs (incl nan), if any ct < thresh,
            mark rows for deletion
        -- if any of the non-nan values below thresh, DELETE COLUMN


    Parameters
    ----------
    _threshold:
        int - the minimum threshold frequency for this column
    _nan_key:
        Union[float, str, Literal[False]] - the nan value found in the
        column in its original dtype. as of 25_01, _columns_getter is
        converting all nan-like values to numpy.nan.
    _nan_ct:
        Union[int, Literal[False]] - the number of nan-like values found
        in the column.
    _COLUMN_UNQ_CT_DICT:
        dict[DataType, int] - the value from _total_cts_by_column for
        this column which is a dictionary containing the uniques and
        their frequencies. cannot be empty.


    Return
    ------
    -
        _instr_list: list[Union[str, DataType]] - the row and column
        operations for this column.


    """

    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    if 'nan' in list(map(str.lower, map(str, _COLUMN_UNQ_CT_DICT.keys()))):
        raise ValueError(f"nan-like is in _UNQ_CTS_DICT and should not be")

    if len(_COLUMN_UNQ_CT_DICT) != 2:
        raise ValueError(f"len(_UNQ_CTS_DICT) != 2")

    if (_nan_ct is False) + (_nan_key is False) not in [0, 2]:
        raise ValueError(f"_nan_key is {_nan_key} and _nan_ct is {_nan_ct}")

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # nans should not be in _COLUMN_UNQ_CT_DICT!

    _instr_list = []
    if not _nan_ct:  # EITHER IGNORING NANS OR NONE IN FEATURE

        _ctr = 0
        for unq, ct in _COLUMN_UNQ_CT_DICT.items():
            if ct < _threshold:
                _ctr += 1
                _instr_list.append(unq)
        if _ctr > 0:
            _instr_list.append('DELETE COLUMN')
        del _ctr, unq, ct


    else:  # HAS NANS AND NOT IGNORING

        _ctr = 0
        for unq, ct in _COLUMN_UNQ_CT_DICT.items():
            if ct < _threshold:
                _ctr += 1
                _instr_list.append(unq)

        if _nan_ct < _threshold:
            _instr_list.append(_nan_key)

        if _ctr > 0:
            _instr_list.append('DELETE COLUMN')
        del _ctr, unq, ct


    return _instr_list








