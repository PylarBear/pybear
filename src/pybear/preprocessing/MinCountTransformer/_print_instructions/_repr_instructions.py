# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence
from typing_extensions import Union
from .._type_aliases import InstructionsType, TotalCountsByColumnType

from copy import deepcopy
import numpy as np

from .._make_instructions._validation._delete_instr import _val_delete_instr
from .._make_instructions._validation._total_counts_by_column import \
    _val_total_counts_by_column
from .._validation._count_threshold import _val_count_threshold
from .._validation._feature_names_in import _val_feature_names_in

from .._make_instructions._threshold_listifier import _threshold_listifier



def _repr_instructions(
    _delete_instr: InstructionsType,
    _total_counts_by_column: TotalCountsByColumnType,
    _thresholds: Union[int, Sequence[int]],
    _n_features_in: int,
    _feature_names_in: Union[Sequence[str], None],
    _clean_printout: bool,
    _max_print_len: int
):

    """
    Display instructions generated for the current fitted state, subject
    to the current settings of the parameters. The printout will indicate
    what values and columns will be deleted, and if all columns or all
    rows will be deleted. Use :method: set_params after finishing fits
    to change MCTs parameters and see the impact on the transformation.

    If the instance has multiple recursions (i.e., :param: max_recursions
    is > 1, parameters cannot be changed via :method: set_params, but
    the net effect of the actual transformation that was performed is
    displayed (remember that multiple recursions can only be accessed
    through :method: fit_transform). The results are displayed as a
    single set of instructions, as if to perform the cumulative effect
    of the recursions in a single step.


    Parameters
    ----------
    _delete_instr:
        dict[
            int,
            list[any, Literal['INACTIVE', 'DELETE ALL', 'DELETE COLUMN']]
        ] - the instructions for deleting values and columns as generated
        by _make_instructions() from the uniques / counts in
        _total_counts_by_column and the parameter settings.
    _total_counts_by_column:
        dict[int, dict[any, int]] - the uniques and their counts for
        each column in the data.
    _thresholds:
        Union[int, Sequence[int]] - the threshold value(s) that determine
        whether a unique value is removed from a dataset.
    _n_features_in:
        int - the number of features in the data.
    _feature_names_in:
        Union[Sequence[str], None] - the features names of the data if
        the data was passed in a container that had features names, like
        a pandas dataframe. Otherwise, None.
    _clean_printout:
        bool - Truncate printout to fit on screen.
    _max_print_len:
        int - the maximum number of characters to display per line.


    Return
    ------
    -
        None


    """


    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # _n_features_in is validated by _val_delete_instr

    _val_feature_names_in(_feature_names_in)

    _val_delete_instr(_delete_instr, _n_features_in)

    _val_total_counts_by_column(_total_counts_by_column)

    # must be list[int] coming into here
    _val_count_threshold(_thresholds, ['int', 'Sequence[int]'], _n_features_in)

    if not isinstance(_clean_printout, bool):
        raise TypeError(f"'_clean_printout' must be boolean")

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    _thresholds = _threshold_listifier(
        _n_features_in,
        deepcopy(_thresholds)
    )

    _max_print_len = 80
    _pad = 2   # number of spaces between name/thresh info & delete info

    if _feature_names_in is None:
        _feature_names_in = [f"Column {i}" for i in range(_n_features_in)]

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
            if len(_wip_instr) == 0:  # then 'DELETE COLUMN' was the only entry
                print(_cdm)
                continue
        else:
            _all_columns_deleted = False

        if len(_wip_instr) == 0:
            print("No operations.")
            continue

        if _wip_instr[0] == 'INACTIVE':
            print("Ignored.")
            continue

        if 'DELETE ALL' in _wip_instr \
                or len(_wip_instr) == len(_total_counts_by_column[col_idx]):
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

    print(f"\n*** NOTE *** ")
    print(
        f"This print utility can only report the instructions and "
        f"outcomes that can be directly inferred from the information "
        f"learned about uniques and counts during fitting. \nIt cannot "
        f"predict any interaction effects that occur during transform of "
        f"a dataset that may ultimately cause all rows to be deleted. "
        f"\nIt also cannot capture the effects of previously unseen "
        f"values that may be passed during transform."
    )



    # pizza
    # v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
    # OLD REPR FROM _test_threshold. KEEP THIS FOR REFERENCE.
    #
    # __make_instructions: Callable[[...], dict[int, list[str, DataType]]]
    # __nfi: int
    # __fni: Union[npt.NDArray[str], None]
    # __tcbc: dict[int, dict[DataType, int]]
    #
    #
    # __count_threshold = deepcopy(_MCTInstance.count_threshold)
    # __max_recursions = _MCTInstance.max_recursions
    # __make_instructions = _MCTInstance._make_instructions
    # __nfi = _MCTInstance.n_features_in_
    # __fni = getattr(_MCTInstance, 'feature_names_in_', None)
    # __tcbc = _MCTInstance._total_counts_by_column
    #
    #
    # _val_count_threshold(
    #     _threshold,
    #     ['int', 'Sequence[int]'],
    #     __nfi
    # )
    #
    # # __count_threshold must be good was validated in _MCTInstance
    #
    # # if ct_thresh is int, map to full list. make_instructions()
    # # will do this on its own, this is for display purposes.
    # __count_threshold = \
    #     _threshold_listifier(
    #         __nfi,
    #         __count_threshold
    #     )
    #
    #
    # if _threshold is None:
    #     _threshold = deepcopy(__count_threshold)
    # else:
    #     _threshold = \
    #         _threshold_listifier(
    #             __nfi,
    #             _threshold
    #         )
    #
    #
    # if __max_recursions > 1 and not np.array_equal(_threshold, __count_threshold):
    #     raise ValueError(
    #         f"can only test the original threshold when max_recursions > 1."
    #     )
    #
    #
    # _delete_instr = __make_instructions(_threshold=_threshold)
    #
    #
    # if __fni is not None:
    #     _pad = min(25, max(map(len, __fni)))
    # else:
    #     _pad = len(str(f"Column {__nfi}"))
    #
    # _thresh_pad = min(5, max(map(len, map(str, _threshold))))
    #
    # _all_rows_deleted = False
    # ALL_COLUMNS_DELETED = []
    # _ardm = f"\nAll rows will be deleted. "  # all_rows_deleted_msg
    # for col_idx, _instr in _delete_instr.items():
    #     _ardd = False  # all_rows_deleted_dummy
    #     if __fni is not None:
    #         _column_name = __fni[col_idx]
    #     else:
    #         _column_name = f"Column {col_idx + 1}"
    #
    #     if len(_instr) == 0:
    #         print(
    #             f"{_column_name[:_pad]}".ljust(_pad + 5),
    #             str(_threshold[col_idx]).ljust(_thresh_pad + 2),
    #             "No operations."
    #         )
    #         continue
    #
    #     if _instr[0] == 'INACTIVE':
    #         print(
    #             f"{_column_name[:_pad]}".ljust(_pad + 5),
    #             str(_threshold[col_idx]).ljust(_thresh_pad + 2),
    #             "Ignored"
    #         )
    #         continue
    #
    #     _delete_rows, _delete_col = "", ""
    #     if 'DELETE COLUMN' in _instr:
    #         _delete_col = f"Delete column."
    #         # IF IS BIN-INT & NOT DELETE ROWS, ONLY ENTRY WOULD BE "Delete column."
    #         _instr = _instr[:-1]
    #         ALL_COLUMNS_DELETED.append(True)
    #     else:
    #         ALL_COLUMNS_DELETED.append(False)
    #
    #     if len(_instr) == len(__tcbc[col_idx]):
    #         _ardd = True
    #         _all_rows_deleted = True
    #
    #     if len(_instr):
    #         _delete_rows = "Delete rows associated with "
    #         ctr = 0
    #         for _value in _instr:
    #             try:
    #                 _value = np.float64(str(_value)[:7])
    #                 _delete_rows += f"{_value}"
    #             except:
    #                 _delete_rows += str(_value)
    #             ctr += 1
    #             _max_print_len = (80 - _pad - 5 - _ardd * (len(_ardm) - 20))
    #             if _clean_printout and len(_delete_rows) >= _max_print_len and \
    #                     len(_instr[ctr:]) > 0:
    #                 _delete_rows += f"... + {len(_instr[ctr:])} other(s). "
    #                 break
    #             if len(_instr[ctr:]) > 0:
    #                 _delete_rows += ", "
    #             else:
    #                 _delete_rows += ". "
    #
    #         if _all_rows_deleted:
    #             _delete_rows += _ardm
    #
    #         del ctr, _value
    #
    #     print(
    #         f"{_column_name[:_pad]}".ljust(_pad + 5),
    #         str(_threshold[col_idx]).ljust(_thresh_pad + 2),
    #         _delete_rows + _delete_col
    #     )
    #
    # if False not in ALL_COLUMNS_DELETED:
    #     print(f'\nAll columns will be deleted.')
    # if _all_rows_deleted:
    #     print(f'{_ardm}')
    #
    # del _threshold, _delete_instr, _all_rows_deleted, _ardm, col_idx, _instr
    # del _column_name, _delete_rows, _delete_col, _pad, _ardd, ALL_COLUMNS_DELETED











