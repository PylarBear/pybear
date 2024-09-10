# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from typing_extensions import Union, Literal
from ..._type_aliases import DataType



def _two_uniques_not_hab(
    _instr_list: list,
    _threshold: int,
    _nan_key: Union[float, str, Literal[False]],
    _nan_ct: Union[int, Literal[False]],
    _COLUMN_UNQ_CT_DICT: dict[DataType, int],
    _delete_axis_0: bool,
    _dtype: str
    ) -> list[Union[str, DataType]]:


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
            for deletion if dtype not 'int' or delete_axis_0 is True
        -- if any below thresh, DELETE COLUMN
    - if not ign nan and has nans for non-int or if delete_axis_0 is True
        -- put nan & ct back in dict, treat it like any other value
        -- look at cts for the 3 unqs, if any ct < thresh, mark rows
            for deletion
        -- if any of the non-nan values below thresh, DELETE COLUMN
    - if not ign nan and has nans for int and not delete_axis_0
        -- look at cts for the 2 unqs, if any ct < thresh, DELETE COLUMN
        -- but if keeping the column (both above thresh) and nan ct
            less than thresh, delete the nans

    Parameters
    ----------
    _instr_list: list, should be empty
    _threshold: int
    _nan_key: Union[float, str, Literal[False]]
    _nan_ct: Union[int, Literal[False]]
    _COLUMN_UNQ_CT_DICT: dict[DataType, int], cannot be empty
    _delete_axis_0: bool
    _dtype: str

    Return
    ------
    -
        _instr_list: list[Union[str, DataType]]


    """

    if not len(_instr_list) == 0:
        raise ValueError(f"'_instr_list' must be empty")

    if 'nan' in list(map(str.lower, map(str, _COLUMN_UNQ_CT_DICT.keys()))):
        raise ValueError(f"nan-like is in _UNQ_CTS_DICT and should not be")

    if len(_COLUMN_UNQ_CT_DICT) != 2:
        raise ValueError(f"len(_UNQ_CTS_DICT) != 2")

    if (_nan_ct is False) + (_nan_key is False) not in [0, 2]:
        raise ValueError(f"_nan_key is {_nan_key} and _nan_ct is {_nan_ct}")


    if not _nan_ct:  # EITHER IGNORING NANS OR NONE IN FEATURE

        _ctr = 0
        for unq, ct in _COLUMN_UNQ_CT_DICT.items():
            if ct < _threshold:
                _ctr += 1
                if _dtype != 'int' or _delete_axis_0:
                    _instr_list.append(unq)
        if _ctr > 0:
            _instr_list.append('DELETE COLUMN')
        del _ctr, unq, ct


    else:  # HAS NANS AND NOT IGNORING

        if (_dtype != 'int') or _delete_axis_0:
            _COLUMN_UNQ_CT_DICT[_nan_key] = _nan_ct
            non_nan_ctr = 0
            for unq, ct in _COLUMN_UNQ_CT_DICT.items():
                if ct < _threshold:
                    if str(unq).lower() != 'nan':
                        non_nan_ctr += 1
                    _instr_list.append(unq)
            if non_nan_ctr > 0:
                _instr_list.append('DELETE COLUMN')
            del non_nan_ctr, unq, ct
        else:  # elif _dtype == 'int' and not delete_axis_0
            # nan IS NOT PUT BACK IN
            if min(list(_COLUMN_UNQ_CT_DICT.values())) < _threshold:
                _instr_list.append('DELETE COLUMN')
            elif _nan_ct < _threshold:
                _instr_list.append(_nan_key)



    return _instr_list








