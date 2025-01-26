# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union, Literal
from .._type_aliases import DataType

import numpy as np



def _two_uniques_hab(
    _instr_list: list,
    _threshold: int,
    _nan_key: Union[float, str, Literal[False]],
    _nan_ct: Union[int, Literal[False]],
    _COLUMN_UNQ_CT_DICT: dict[DataType, int],
    _delete_axis_0: bool
) -> list[Union[Literal['DELETE COLUMN'], DataType]]:


    """

    Make delete instructions for a column with two unique non-nan values,
    handling values as booleans.

    WHEN 2 items (NOT INCLUDING nan):
    *** BINARY INT COLUMNS ARE HANDLED DIFFERENTLY THAN OTHER DTYPES ***
    Most importantly, if a binary integer has a value below threshold,
    the DEFAULT behavior is to not delete the respective rows (whereas
    any other dtype will delete the rows), essentially causing bin int
    columns with insufficient count to just be deleted.


    - classify uniques into two classes - 'zero' and 'non-zero'

    - if ignoring nan or no nans
        -- look at cts for the 2 classes, if any ct < thresh, mark all
            associated values for deletion if delete_axis_0 is True
        -- if any class below thresh, DELETE COLUMN
    - if not ign nan and has nans if delete_axis_0 is True
        -- treat nan like any other value
        -- look at cts for the 3 classes, if any ct < thresh, mark all
            associated values for deletion
        -- if any of the non-nan classes below thresh, DELETE COLUMN
    - if not ign nan and has nans if not delete_axis_0
        -- look at cts for the 2 non-nan classes, if any ct < thresh,
            DELETE COLUMN
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

    if any([isinstance(x, str) for x in _COLUMN_UNQ_CT_DICT]):
        raise TypeError(f"handle_as_bool on a str column")


    # no nans should be in _COLUMN_UNQ_CT_DICT!

    # SINCE HANDLING AS BOOL, ONLY NEED TO KNOW WHAT IS NON-ZERO AND
    # IF ROWS WILL BE DELETED OR KEPT
    UNQS = np.fromiter(_COLUMN_UNQ_CT_DICT.keys(), dtype=np.float64)
    CTS = np.fromiter(_COLUMN_UNQ_CT_DICT.values(), dtype=np.float64)
    NON_ZERO_MASK = UNQS.astype(bool)
    NON_ZERO_UNQS = UNQS[NON_ZERO_MASK]
    total_non_zeros = CTS[NON_ZERO_MASK].sum()
    total_zeros = CTS[np.logical_not(NON_ZERO_MASK)].sum()
    del UNQS, CTS, NON_ZERO_MASK


    if not _nan_ct:  # EITHER IGNORING NANS OR NONE IN FEATURE

        if _delete_axis_0:
            if total_zeros and total_zeros < _threshold:
                _instr_list.append(0)

            if total_non_zeros and total_non_zeros < _threshold:
                for k in NON_ZERO_UNQS:
                    _instr_list.append(k)
                del k

        if (total_zeros < _threshold) or (total_non_zeros < _threshold):
            _instr_list.append('DELETE COLUMN')


    else:  # HAS NANS AND NOT IGNORING

        if _delete_axis_0:
            if total_zeros and total_zeros < _threshold:
                _instr_list.append(0)

            if total_non_zeros and total_non_zeros < _threshold:
                for k in NON_ZERO_UNQS:
                    _instr_list.append(k)
                del k

            if _nan_ct and _nan_ct < _threshold:
                _instr_list.append(_nan_key)

            if (total_zeros < _threshold) or (total_non_zeros < _threshold):
                _instr_list.append('DELETE COLUMN')

        elif not _delete_axis_0:
            # only delete nans if below threshold and not deleting column
            # OTHERWISE IF _nan_ct < _threshold but not delete_axis_0
            # AND NOT DELETE COLUMN THEY WOULD BE KEPT DESPITE
            # BREAKING THRESHOLD
            if (total_zeros < _threshold) or (total_non_zeros < _threshold):
                _instr_list.append('DELETE COLUMN')
            elif _nan_ct and _nan_ct < _threshold:
                _instr_list.append(_nan_key)


    del NON_ZERO_UNQS, total_zeros, total_non_zeros

    return _instr_list








