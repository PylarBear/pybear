# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Callable, Iterable
from typing_extensions import Union
import numpy.typing as npt
from ._type_aliases import DataType

import numbers
import numpy as np

from ._validation._count_threshold import _val_count_threshold



def _test_threshold(
    _MCTInstance,
    _threshold: Union[int, Iterable[int], None],
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
        int - count_threshold value(s) to test.
    _clean_printout:
        bool - Truncate printout to fit on screen.


    Return
    ------
    -
        None

    """


    __make_instructions: Callable[[...], dict[int, list[str, DataType]]]
    __nfi: int
    __fni: Union[npt.NDArray[str], None]
    __tcbc: dict[int, dict[DataType, int]]


    __count_threshold = _MCTInstance.count_threshold
    __max_recursions = _MCTInstance.max_recursions
    __make_instructions = _MCTInstance._make_instructions
    __nfi = _MCTInstance.n_features_in_
    __fni = getattr(_MCTInstance, 'feature_names_in_', None)
    __tcbc = _MCTInstance._total_counts_by_column



    if _threshold is None:
        _threshold = __count_threshold
        # OK FOR 1 OR MORE RECURSIONS
        if isinstance(_threshold, numbers.Integral):
            _threshold = [_threshold for _ in range(__nfi)]
    else:
        _val_count_threshold(
            _threshold,
            ['int', 'Iterable[int]'],
            __nfi
        )

        # if ct_thresh is int, map to full list. make_instruction()
        # will do this on its own, this is for display purposes.
        if isinstance(_threshold, numbers.Integral):
            _threshold = [_threshold for _ in range(__nfi)]

        if isinstance(__count_threshold, numbers.Integral):
            __count_threshold = [__count_threshold for _ in range(__nfi)]

        if __max_recursions == 1:
            if np.any((np.array(_threshold) < 2)):
                raise ValueError(f"threshold(s) must be integer >= 2. ")
        elif __max_recursions > 1:
            if not np.array_equal(_threshold, __count_threshold):
                raise ValueError(
                    f"can only test the original threshold "
                    f"when max_recursions > 1. "
                )


    _delete_instr = __make_instructions(_threshold=_threshold)

    # old _delete_instr repr v v v v v v v v v v v v v v v v v v v v v v v v
    if __fni is not None:
        _pad = min(25, max(map(len, __fni)))
    else:
        _pad = len(str(f"Column {__nfi}"))

    _thresh_pad = min(5, max(map(len, map(str, _threshold))))

    _all_rows_deleted = False
    ALL_COLUMNS_DELETED = []
    _ardm = f"\nAll rows will be deleted. "  # all_rows_deleted_msg
    for col_idx, _instr in _delete_instr.items():
        _ardd = False  # all_rows_deleted_dummy
        if __fni is not None:
            _column_name = __fni[col_idx]
        else:
            _column_name = f"Column {col_idx + 1}"

        if len(_instr) == 0:
            print(f"{_column_name[:_pad]}".ljust(_pad + 5), str(_threshold[col_idx]).ljust(_thresh_pad + 2), "No operations.")
            continue

        if _instr[0] == 'INACTIVE':
            print(f"{_column_name[:_pad]}".ljust(_pad + 5), str(_threshold[col_idx]).ljust(_thresh_pad + 2), "Ignored")
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

        print(f"{_column_name[:_pad]}".ljust(_pad + 5), str(_threshold[col_idx]).ljust(_thresh_pad + 2), _delete_rows + _delete_col)

    if False not in ALL_COLUMNS_DELETED:
        print(f'\nAll columns will be deleted.')
    if _all_rows_deleted:
        print(f'{_ardm}')

    del _threshold, _delete_instr, _all_rows_deleted, _ardm, col_idx, _instr
    del _column_name, _delete_rows, _delete_col, _pad, _ardd, ALL_COLUMNS_DELETED




















