# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Iterable
from typing_extensions import Union
from .._type_aliases import InstructionsType, TotalCountsByColumnType

from copy import deepcopy
import numpy as np

from ._validation._delete_instr import _val_delete_instr
from ._validation._total_counts_by_column import _val_total_counts_by_column
from .._validation._count_threshold import _val_count_threshold
from .._validation._feature_names_in import _val_feature_names_in




def _repr_instructions(
    _delete_instr: InstructionsType,
    _total_counts_by_column: TotalCountsByColumnType,
    _thresholds: Iterable[int],
    _n_features_in: int,
    _feature_names_in: Union[Iterable[str], None],
    # Pizza put _max_print_len in here!.... and do validation
    _clean_printout: bool
):

    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # _n_features_in is validated by _val_delete_instr

    _val_feature_names_in(_feature_names_in)

    _val_delete_instr(_delete_instr, _n_features_in)

    _val_total_counts_by_column(_total_counts_by_column)

    # must be list[int] coming into here
    _val_count_threshold(_thresholds, ['Iterable[int]'], _n_features_in)

    if not isinstance(_clean_printout, bool):
        raise TypeError(f"'_clean_printout' must be boolean")

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    _max_print_len = 80
    _pad = 2   # number of spaces between name/thresh info & delete info

    _tcnw = min(25, max(map(len, _feature_names_in)))
    # total column name width  ... the total number of spaces allowed
    # for f"column name (threshold)" and _pad.
    _tcnw += (max(map(len, map(str, _thresholds))) + 3)
    _tcnw += _pad

    _delete_column = False
    _all_rows_deleted = False
    _all_columns_deleted = True  # if any column is not deleted, toggles to False
    _ardm = f"All rows will be deleted. "  # all_rows_deleted_msg
    _cdm = f"Delete column."   # column_delete_msg
    for col_idx, _instr in _delete_instr.items():

        _wip_instr = deepcopy(_instr)

        # print the column & threshold part -- -- -- -- -- -- -- -- -- --
        if _feature_names_in is not None:
            _column_name = _feature_names_in[col_idx]
        else:
            _column_name = f"Column {col_idx + 1}"

        _col_w = len(_column_name)
        _thresh_w = len(f" ({_thresholds[col_idx]})")

        if (_col_w + _thresh_w) > (_tcnw - _pad):
            _trunc_len = _tcnw - _pad - _thresh_w - 4  # (-4 for space/ellipsis)
            _column_name = _column_name[:_trunc_len] + f"..."
            del _trunc_len

        del _col_w, _thresh_w

        _base_printout = f"{_column_name} ({_thresholds[col_idx]})"

        assert len(_base_printout) <= _tcnw

        # notice the end="" here!
        print(_base_printout.ljust(_tcnw), end="")

        del _column_name, _base_printout   # dont delete _tcnw
        # END print the column & threshold part -- -- -- -- -- -- -- --

        # print the instructions part -- -- -- -- -- -- -- -- -- -- --

        if 'DELETE COLUMN' in _wip_instr:
            # IF IS BIN-INT & NOT DELETE ROWS, ONLY ENTRY WOULD BE "DELETE COLUMN"
            _delete_column = True
            _wip_instr = _wip_instr[:-1]
        else:
            _all_columns_deleted = False

        if len(_wip_instr) == 0:
            print("No operations.")
            continue

        if _wip_instr[0] == 'INACTIVE':
            print("Ignored.")
            continue

        # do not skip to next here if 'DELETE COLUMN' is the only entry.
        # need to go thru the for loop to see if all columns are deleted.

        if len(_wip_instr) == len(_total_counts_by_column[col_idx]):
            _all_rows_deleted = True

        # condition the values in _wip_instr for easier viewing
        for _idx, _value in enumerate(_wip_instr):
            try:
                _value = np.float64(str(_value)[:7])
                _wip_instr[_idx] = f"{_value}"
            except:
                _wip_instr[_idx] = str(_value)

        _wip_instr: list[str]

        _delete_rows_msg = "Delete rows containing "
        _mpl = _max_print_len - _tcnw   # subtract allowance for column name
        _mpl -= len(_delete_rows_msg)   # subtract the row del prefix jargon
        _mpl -= len(_ardm) if _all_rows_deleted else 0 # subtract all row del
        _mpl -= len(_cdm) if _delete_column else 0 # subtract col del jargon
        # what is left in _mpl is the num spaces we have to put row values


        trunc_msg = lambda _idx: f"... + {len(_wip_instr[_idx:])} other(s). "

        if not _clean_printout or len(', '.join(_wip_instr) + ". ") <= _mpl:
            _delete_rows_msg += (', '.join(_wip_instr) + ". ")
            # if the total length of the entries is less than _mpl, just
            # join and get on with it
        elif (_mpl - len(trunc_msg(0))) <= 0:
            _delete_rows_msg = f"{len(_wip_instr)} unique values will be deleted. "
        else:
            _running_len_tally = 0
            for _idx, _value in enumerate(_wip_instr):
                _running_len_tally += len(_value)
                if (_running_len_tally + len(trunc_msg(_idx))) >= _mpl:
                    break
            # keep this _idx number!

            _delete_rows_msg += ", ".join(_wip_instr[:_idx])
            _delete_rows_msg += f"... + {len(_wip_instr[_idx:])} other(s). "


        _delete_rows_msg += _ardm if _all_rows_deleted else ""

        _delete_rows_msg += _cdm if _delete_column else ""

        # pizza
        # assert len(_delete_rows_msg) <= (_max_print_len - _tcnw)

        print(_delete_rows_msg)

    if _all_columns_deleted:
        print(f'\nAll columns will be deleted.')
    if _all_rows_deleted:
        print(f'\nAll rows are guaranteed to be deleted.')















