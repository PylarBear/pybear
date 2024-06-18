# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing import Union, Callable
from ._type_aliases import DataType


import numpy as np




def _test_threshold(
    _MCTInstance,
    _threshold: Union[int, None],
    _clean_printout: bool
) -> None:

    """
    Display instructions generated for the current fitted state, subject
    to the passed threshold and the current settings of other parameters.
    The printout will indicate what rows / columns will be deleted, and
    if all columns or all rows will be deleted.

    If the instance has multiple recursions, the results are displayed
    as a single set of instructions, as if to perform the cumulative
    effect of the recursions in a single step.

    Parameters
    ----------
    _MCTInstance:
        the instance of MinCountTransformer
    _threshold:
        int - count_threshold value to test.
    _clean_printout:
        bool - Truncate printout to fit on screen.

    Return
    ------
    -
        None

    """


    __make_instructions: Callable[[...], dict[int: list[str, DataType]]]
    __nfi: int
    __fni: Union[bool, np.ndarray[str]]
    __tcbc: dict[int, dict[DataType, int]]


    __count_threshold = _MCTInstance._count_threshold
    __max_recursions = _MCTInstance._max_recursions
    __make_instructions = _MCTInstance._make_instructions
    __nfi = _MCTInstance.n_features_in_
    __fni = False
    if hasattr(_MCTInstance, 'feature_names_in_'):
        __fni = _MCTInstance.feature_names_in_
    __tcbc = _MCTInstance._total_counts_by_column


    if _threshold is None:
        _threshold = __count_threshold
        # OK FOR 1 OR MORE RECURSIONS
    else:
        if __max_recursions == 1:
            if int(_threshold) != _threshold or not _threshold >= 2:
                raise ValueError(f"threshold must be an integer >= 2")
        elif __max_recursions > 1:
            if _threshold != __count_threshold:
                raise ValueError(f"can only test the original threshold "
                                 f"when max_recursions > 1")

    _delete_instr = __make_instructions(_threshold=_threshold)

    print(f'\nThreshold = {_threshold}')

    if __fni is not False:
        _pad = min(25, max(map(len, __fni)))
    else:
        _pad = len(str(f"Column {__nfi}"))

    _all_rows_deleted = False
    ALL_COLUMNS_DELETED = []
    _ardm = f"\nAll rows will be deleted."  # all_rows_deleted_msg
    for col_idx, _instr in _delete_instr.items():
        _ardd = False  # all_rows_deleted_dummy
        if __fni is not False:
            _column_name = __fni[col_idx]
        else:
            _column_name = f"Column {col_idx + 1}"

        if len(_instr) == 0:
            print(f"{_column_name[:_pad]}".ljust(_pad + 5), "No operations.")
            continue

        if _instr[0] == 'INACTIVE':
            print(f"{_column_name[:_pad]}".ljust(_pad + 5), "Ignored")
            continue

        _delete_rows, _delete_col = "", ""
        if 'DELETE COLUMN' in _instr:
            _delete_col = f"Delete column."
            # IF IS BIN-INT & NOT DELETE ROWS, ONLY ENTRY WOULD BE "Delete column."
            _instr = _instr[:-1]
            ALL_COLUMNS_DELETED.append(True)
        else:
            ALL_COLUMNS_DELETED.append(False)

        if len(_instr) == len(__tcbc[col_idx]):
            _ardd = True
            _all_rows_deleted = True

        if len(_instr):
            _delete_rows = "Delete rows associated with "
            ctr = 0
            for _value in _instr:
                try:
                    _value = np.float64(str(_value)[:7])
                    _delete_rows += f"{_value}"
                except:
                    _delete_rows += str(_value)
                ctr += 1
                _max_print_len = (80 - _pad - 5 - _ardd * (len(_ardm) - 20))
                if _clean_printout and len(_delete_rows) >= _max_print_len and \
                        len(_instr[ctr:]) > 0:
                    _delete_rows += f"... + {len(_instr[ctr:])} other(s). "
                    break
                if len(_instr[ctr:]) > 0:
                    _delete_rows += ", "
                else:
                    _delete_rows += ". "

            if _all_rows_deleted:
                _delete_rows += _ardm

            del ctr, _value

        print(f"{_column_name[:_pad]}".ljust(_pad + 5), _delete_rows + _delete_col)

    if False not in ALL_COLUMNS_DELETED:
        print(f'\nAll columns will be deleted.')
    if _all_rows_deleted:
        print(f'{_ardm}')

    del _threshold, _delete_instr, _all_rows_deleted, _ardm, col_idx, _instr
    del _column_name, _delete_rows, _delete_col, _pad, _ardd, ALL_COLUMNS_DELETED





















