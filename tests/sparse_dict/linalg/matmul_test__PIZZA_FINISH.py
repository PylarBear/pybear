# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import numpy as np
from pybear.sparse_dict import _validation as val

from pybear.sparse_dict._utils import (
                                        outer_len,
                                        inner_len,
                                        inner_len_quick
)


def test_new_core_matmul(DICT1, DICT2, DICT2_TRANSPOSE=None, return_as=None):
    '''DICT1 and DICT2 enter oriented with sparse inner dicts as rows, with multiplication rules like NumPy.  There is
        no protection here to prevent dissimilar sized rows from DICT1 dotting with columns from DICT2.
        Create posn for last entry, so that placeholder rules are enforced (original length of object is retained).'''

    # Transpose DICT2 for ease of multiplication, not for matmul rules
    if not DICT2_TRANSPOSE is None:
        DICT2_T = DICT2_TRANSPOSE
    else:
        DICT2_T = sparse_transpose(DICT2)

    if not return_as is None: return_as = return_as.upper()

    if return_as is None or return_as == 'SPARSE_DICT':
        OUTPUT = {int(_): {} for _ in range(outer_len(DICT1))}
    elif return_as == 'ARRAY':
        OUTPUT = np.zeros((outer_len(DICT1), outer_len(DICT2_T)), dtype=float)

    for outer_dict_idx1 in DICT1:
        for outer_dict_idx2 in DICT2_T:
            dot = 0
            # for inner_dict_idx in DICT1[outer_dict_idx1]:
            #     if inner_dict_idx in DICT2_T[outer_dict_idx2]:
            #         dot += DICT1[outer_dict_idx1][inner_dict_idx] * DICT2_T[outer_dict_idx2][inner_dict_idx]
            # for idx in list(set(DICT1[outer_dict_idx1]).intersection(DICT2_T[outer_dict_idx2])):  # SPEED TEST 9/20/22 SUGGEST PULLS AWAY FROM OLD WAY AS SIZES GET BIGGER
            #     dot += DICT1[outer_dict_idx1][idx] * DICT2_T[outer_dict_idx2][idx]

            MASK = set(DICT1[outer_dict_idx1]).intersection(
                DICT2_T[outer_dict_idx2])

            dot = np.matmul(
                np.fromiter((DICT1[outer_dict_idx1][_] for _ in MASK),
                            dtype=np.float64),
                np.fromiter(
                    (np.array(DICT2_T[outer_dict_idx2][_]) for _ in MASK),
                    dtype=np.float64))

            if dot != 0: OUTPUT[int(outer_dict_idx1)][
                int(outer_dict_idx2)] = dot

    if return_as is None or return_as == 'SPARSE_DICT':
        last_inner_idx = outer_len(DICT2_T) - 1
        for outer_dict_idx1 in OUTPUT:
            if last_inner_idx not in OUTPUT[outer_dict_idx1]:
                OUTPUT[int(outer_dict_idx1)][int(last_inner_idx)] = 0

    return OUTPUT


def core_matmul(DICT1, DICT2, DICT2_TRANSPOSE=None, return_as=None):
    '''DICT1 and DICT2 enter oriented with sparse inner dicts as rows, with multiplication rules like NumPy.  There is
        no protection here to prevent dissimilar sized rows from DICT1 dotting with columns from DICT2.
        Create posn for last entry, so that placeholder rules are enforced (original length of object is retained).'''

    # Transpose DICT2 for ease of multiplication, not for matmul rules
    if not DICT2_TRANSPOSE is None:
        DICT2_T = DICT2_TRANSPOSE
    else:
        DICT2_T = sparse_transpose(DICT2)

    if not return_as is None: return_as = return_as.upper()

    if return_as is None or return_as == 'SPARSE_DICT':
        OUTPUT = {int(_): {} for _ in range(outer_len(DICT1))}
    elif return_as == 'ARRAY':
        OUTPUT = np.zeros((outer_len(DICT1), outer_len(DICT2_T)), dtype=float)

    for outer_dict_idx1 in DICT1:
        for outer_dict_idx2 in DICT2_T:
            dot = 0
            # for inner_dict_idx in DICT1[outer_dict_idx1]:
            #     if inner_dict_idx in DICT2_T[outer_dict_idx2]:
            #         dot += DICT1[outer_dict_idx1][inner_dict_idx] * DICT2_T[outer_dict_idx2][inner_dict_idx]
            for idx in set(DICT1[outer_dict_idx1]).intersection(DICT2_T[
                                                                    outer_dict_idx2]):  # SPEED TEST 9/20/22 SUGGEST PULLS AWAY FROM OLD WAY AS SIZES GET BIGGER
                dot += DICT1[outer_dict_idx1][idx] * DICT2_T[outer_dict_idx2][
                    idx]
            if dot != 0: OUTPUT[int(outer_dict_idx1)][
                int(outer_dict_idx2)] = dot

    if return_as is None or return_as == 'SPARSE_DICT':
        last_inner_idx = outer_len(DICT2_T) - 1
        for outer_dict_idx1 in OUTPUT:
            if last_inner_idx not in OUTPUT[outer_dict_idx1]:
                OUTPUT[int(outer_dict_idx1)][int(last_inner_idx)] = 0

    return OUTPUT


def core_symmetric_matmul(DICT1, DICT2, DICT2_TRANSPOSE=None, return_as=None):
    '''DICT1 and DICT2 enter oriented with sparse inner dicts as rows, with multiplication rules like NumPy.
    For use on things like ATA and AAT to save time.  There is no protection here to prevent dissimilar sized
    rows from DICT1 dotting with columns from DICT2.  Create posn for last entry, so that placeholder rules are
    enforced (original length of object is retained).'''

    _display_time = lambda t0: round((time.time() - t0), 2)

    # ********** Transpose DICT2 for ease of multiplication, not for matmul rules.***************************
    print(f'                PIZZA TRANSPOSING DICT2 IN core_symmetric_matmul');
    t0 = time.time()
    if not DICT2_TRANSPOSE is None:
        DICT2_T = DICT2_TRANSPOSE
    else:
        DICT2_T = core_sparse_transpose(DICT2)
    print(
        f'                END PIZZA TRANSPOSING DICT2 IN core_symmetric_matmul. time = {_display_time(t0)}\n')

    _outer_len = outer_len(DICT1)  # == outer_len(DICT2_T) == inner_len(DICT2)
    _inner_len = inner_len(DICT1)

    if not return_as is None: return_as = return_as.upper()

    # MUST CREATE "OUTPUT" BEFORE RUNNING, TO HAVE ALL SLOTS AVAILABLE FOR FILLING
    if return_as is None or return_as == 'SPARSE_DICT':
        OUTPUT = {int(outer_key): {} for outer_key in range(_outer_len)}
    elif return_as == 'ARRAY':
        OUTPUT = np.zeros((_outer_len, _outer_len), dtype=float)  # SYMMETRIC, REMEMBER

    print(
        f'                PIZZA GOING INTO BUILD MATRIX IN core_symmetric_matmul');
    t0 = time.time()
    for outer_dict_idx2 in range(_outer_len):
        for outer_dict_idx1 in range(
                outer_dict_idx2 + 1):  # MUST GET DIAGONAL, SO MUST GET WHERE outer1 = outer2
            # ********* THIS IS THE STEP THAT IS TAKING FOREVER AND JACKING RAM!! ***********
            dot = 0
            # for inner_dict_idx1 in DICT1[outer_dict_idx1]:  # OLD WAY---DOT BY KEY SEARCH
            #     if inner_dict_idx1 in DICT2_T[outer_dict_idx2]:
            #         dot += DICT1[outer_dict_idx1][inner_dict_idx1] * DICT2_T[outer_dict_idx2][inner_dict_idx1]
            for idx in set(DICT1[outer_dict_idx1]).intersection(DICT2_T[
                                                                    outer_dict_idx2]):  # SPEED TEST 9/20/22 SUGGEST PULLS AWAY FROM OLD WAY AS SIZES GET BIGGER
                dot += DICT1[outer_dict_idx1][idx] * DICT2_T[outer_dict_idx2][
                    idx]
            if dot != 0:
                OUTPUT[int(outer_dict_idx2)][int(outer_dict_idx1)] = dot
                OUTPUT[int(outer_dict_idx1)][int(outer_dict_idx2)] = dot

    # CLEAN UP PLACEHOLDER RULES
    if return_as is None or return_as == 'SPARSE_DICT':
        for outer_key in OUTPUT:
            if _outer_len - 1 not in OUTPUT[
                outer_key]:  # OUTER LEN = INNER LEN
                OUTPUT[int(outer_key)][int(_outer_len - 1)] = 0

    print(
        f'                PIZZA DONE BUILD MATRIX IN core_symmetric_matmul. time = {_display_time(t0)}\n')

    return OUTPUT


def matmul(DICT1, DICT2, DICT2_TRANSPOSE=None, return_as=None):
    '''Run matmul with safeguards that assure matrix multiplication rules are followed when running core_matmul().'''

    val._insufficient_dict_args_2(DICT1, DICT2)
    DICT1 = val._dict_init(DICT1)
    DICT2 = val._dict_init(DICT2)
    val._broadcast_check(DICT1, DICT2)  # DO THIS BEFORE TRANSPOSING DICT2

    if not DICT2_TRANSPOSE is None:
        DICT2_TRANSPOSE = val._dict_init(DICT2_TRANSPOSE, fxn)
        val._insufficient_dict_args_1(DICT2_TRANSPOSE, fxn)
        DICT2_T = DICT2_TRANSPOSE
    else:
        DICT2_T = sparse_transpose(DICT2)

    return core_matmul(DICT1, DICT2, DICT2_TRANSPOSE=DICT2_T,
                       return_as=return_as)


def symmetric_matmul(DICT1, DICT2, DICT1_TRANSPOSE=None, DICT2_TRANSPOSE=None,
                     return_as=None):
    '''Run matmul with safeguards that assure matrix multiplication rules are followed when running core_symmetric_matmul().'''

    val._insufficient_dict_args_2(DICT1, DICT2)
    DICT1 = val._dict_init(DICT1)
    DICT2 = val._dict_init(DICT2)
    val._broadcast_check(DICT1, DICT2)  # DO THIS BEFORE TRANSPOSING DICT2
    if not DICT1_TRANSPOSE is None:
        DICT1_TRANSPOSE = val._dict_init(DICT1_TRANSPOSE, fxn)
        val._insufficient_dict_args_1(DICT1_TRANSPOSE, fxn)
        DICT1_T = DICT1_TRANSPOSE
    else:
        DICT1_T = sparse_transpose(DICT1)
    if not DICT2_TRANSPOSE is None:
        DICT2_TRANSPOSE = val._dict_init(DICT2_TRANSPOSE, fxn)
        val._insufficient_dict_args_1(DICT2_TRANSPOSE, fxn)
        DICT2_T = DICT2_TRANSPOSE
    else:
        DICT2_T = sparse_transpose(DICT2)

    val._symmetric_matmul_check(DICT1, DICT2, DICT1_TRANSPOSE=DICT1_T,
                           DICT2_TRANSPOSE=DICT2_T)

    return core_symmetric_matmul(DICT1, DICT2, DICT2_TRANSPOSE=DICT2_T,
                                 return_as=return_as)


def sparse_ATA(DICT1, DICT1_TRANSPOSE=None, return_as=None):
    '''Calculates ATA on DICT1 using symmetric matrix multiplication.'''

    val._insufficient_dict_args_1(DICT1)
    DICT1 = val._dict_init(DICT1)

    if not DICT1_TRANSPOSE is None:
        DICT1_TRANSPOSE = val._dict_init(DICT1_TRANSPOSE, fxn)
        val._insufficient_dict_args_1(DICT1_TRANSPOSE, fxn)
        DICT1_T = DICT1_TRANSPOSE
    else:
        DICT1_T = sparse_transpose(DICT1)

    # 9/18/22 CHANGED FROM symmetric_matmul TO core_symmetric_matmul FOR SPEED
    _ = core_symmetric_matmul(DICT1_T, DICT1, DICT2_TRANSPOSE=DICT1_T,
                              return_as=return_as)
    return _


def sparse_AAT(DICT1, DICT1_TRANSPOSE=None, return_as=None):
    '''Calculates AAT on DICT1 using symmetric matrix multiplication.'''

    val._insufficient_dict_args_1(DICT1)
    DICT1 = val._dict_init(DICT1)

    if not DICT1_TRANSPOSE is None:
        DICT1_TRANSPOSE = val._dict_init(DICT1_TRANSPOSE)
        val._insufficient_dict_args_1(DICT1_TRANSPOSE)
        DICT1_T = DICT1_TRANSPOSE
    else:
        DICT1_T = sparse_transpose(DICT1)

    # 9/18/22 CHANGED FROM symmetric_matmul TO core_symmetric_matmul FOR SPEED
    return core_symmetric_matmul(DICT1, DICT1_T, DICT2_TRANSPOSE=DICT1,
                                 return_as=return_as)


def sparse_matmul_from_lists(LIST1, LIST2, LIST2_TRANSPOSE=None,
                             is_symmetric=False):
    '''Calculates matrix product of two lists and returns as sparse dict.'''

    val._insufficient_list_args_1(LIST1)
    val._insufficient_list_args_1(LIST2)
    LIST1 = val._list_init(LIST1)[0]
    LIST2 = val._list_init(LIST2)[0]

    # BROADCAST CHECK
    if len(LIST1[0]) != len(LIST2):
        raise Exception(
            f'{module_name()}.{fxn}() requires for LIST1(m x n) and LIST2(j x k) that num inner keys (n) of\n'
            f'LIST1 == num outer keys (j) of LIST2 ---- (m, n) x (j, k) --> (m, k)\n'
            f'{inner_len(LIST1)} is different than {outer_len(LIST2)}.')

    if not LIST2_TRANSPOSE is None:
        LIST2_TRANSPOSE = val._list_init(LIST2_TRANSPOSE, fxn)
        val._insufficient_list_args_1(LIST2_TRANSPOSE, fxn)
        LIST2_T = LIST2_TRANSPOSE
    else:
        LIST2_T = np.array(LIST2).transpose()

    final_inner_len = len(LIST2_T[0])
    DICT1 = {}
    for outer_idx1 in range(len(LIST1)):
        DICT1[outer_idx1] = {}
        if not is_symmetric:
            inner_iter_end = final_inner_len
        elif is_symmetric:
            inner_iter_end = outer_idx1 + 1  # HAVE TO ITER TO WHERE outer_idx2 == outer_idx1 TO GET DIAGONAL
        for outer_idx2 in range(inner_iter_end):
            dot = np.matmul(LIST1[outer_idx1], LIST2[outer_idx2], dtype=float)
            if dot != 0:
                if not is_symmetric:
                    DICT1[int(outer_idx1)][int(outer_idx2)] = dot
                if is_symmetric:
                    DICT1[int(outer_idx1)][int(outer_idx2)] = dot
                    DICT1[int(outer_idx2)][int(outer_idx1)] = dot

    # PLACEHOLDER RULES
    for outer_key in range(len(DICT1)):
        DICT1[int(outer_key)][int(final_inner_len - 1)] = DICT1[outer_key].get(
            final_inner_len - 1, 0)

    # CHECK outer_len
    return DICT1


def hybrid_matmul(LIST_OR_DICT1, LIST_OR_DICT2, LIST_OR_DICT2_TRANSPOSE=None,
                  return_as='SPARSE_DICT', return_orientation='ROW'):
    '''Left and right object oriented as []=row, with standard numpy matmul and linear algebra rules, and
        safeguards that assure matrix multiplication rules are followed when running core_hybrid_matmul().'''

    return_as = akv.arg_kwarg_validater(
                                        return_as,
                                        'return_as',
                                        ['ARRAY', 'SPARSE_DICT', None],
                                        'sparse_dict',
                                        return_if_none='SPARSE_DICT'
    )

    return_orientation = akv.arg_kwarg_validater(
                                                 return_orientation,
                                                 'return_orientation',
                                                 ['ROW', 'COLUMN', None],
                                                 'sparse_dict',
                                                 return_if_none='ROW'
    )

    if isinstance(LIST_OR_DICT1, dict) and isinstance(LIST_OR_DICT2, dict):
        raise Exception(
            f'{module_name()}.{fxn}() LIST_OR_DICT1 AND LIST_OR_DICT2 CANNOT '
            f'BOTH BE dict. USE sparse_matmul.')

    if isinstance(LIST_OR_DICT1, (np.ndarray, list, tuple)) and isinstance(
            LIST_OR_DICT2, (np.ndarray, list, tuple)):
        raise Exception(
            f'{module_name()}.{fxn}() LIST_OR_DICT1 AND LIST_OR_DICT2 CANNOT '
            f'BOTH BE LIST-TYPE. USE numpy.matmul.')

    if not LIST_OR_DICT2_TRANSPOSE is None and type(LIST_OR_DICT2) != type(
            LIST_OR_DICT2_TRANSPOSE):
        raise Exception(
            f'{module_name()}.{fxn}() IF LIST_OR_DICT2_TRANSPOSE IS GIVEN, '
            f'IT MUST BE THE SAME type AS LIST_OR_DICT2.')

    def dimension_exception(name2, _n, _j): \
            raise Exception(
                f'{module_name()}.{fxn}() requires for LIST_OR_DICT1 (m x n) and {name2} (j x k) that inner length (n) of ' \
                f'LIST_OR_DICT1 == {f"outer" if name2 == "LIST_OR_DICT2" else "inner"} length (j) of {name2} ---- (m, n) x (j, k) --> (m, k).  ' \
                f'n ({_n}) is different than j ({_j}).')

    if isinstance(LIST_OR_DICT1, dict):
        LIST_OR_DICT1 = val._dict_init(LIST_OR_DICT1, fxn)
        val._insufficient_dict_args_1(LIST_OR_DICT1, fxn)

        LIST_OR_DICT2 = val._list_init(np.array(LIST_OR_DICT2), fxn)[0]
        val._insufficient_list_args_1(LIST_OR_DICT2, fxn)

        if LIST_OR_DICT2_TRANSPOSE is None:
            _n, _j = inner_len(LIST_OR_DICT1), len(LIST_OR_DICT2)
            if _n != _j: dimension_exception("LIST_OR_DICT2", _n, _j)
        elif not LIST_OR_DICT2_TRANSPOSE is None:
            LIST_OR_DICT2_TRANSPOSE = \
            val._list_init(np.array(LIST_OR_DICT2_TRANSPOSE), fxn)[0]
            val._insufficient_list_args_1(LIST_OR_DICT2_TRANSPOSE, fxn)
            _n, _j = inner_len(LIST_OR_DICT1), len(LIST_OR_DICT2_TRANSPOSE[0])
            if _n != _j: dimension_exception("LIST_OR_DICT2_TRANSPOSE", _n, _j)
    elif isinstance(LIST_OR_DICT2, dict):
        LIST_OR_DICT1 = val._list_init(np.array(LIST_OR_DICT1), fxn)[0]
        val._insufficient_list_args_1(LIST_OR_DICT1, fxn)

        LIST_OR_DICT2 = val._dict_init(LIST_OR_DICT2, fxn)
        val._insufficient_dict_args_1(LIST_OR_DICT2, fxn)

        if LIST_OR_DICT2_TRANSPOSE is None:
            _n, _j = len(LIST_OR_DICT1[0]), outer_len(LIST_OR_DICT2)
            if _n != _j: dimension_exception("LIST_OR_DICT2", _n, _j)
        elif not LIST_OR_DICT2_TRANSPOSE is None:
            LIST_OR_DICT2_TRANSPOSE = val._dict_init(LIST_OR_DICT2_TRANSPOSE, fxn)
            val._insufficient_dict_args_1(LIST_OR_DICT2_TRANSPOSE, fxn)
            _n, _j = len(LIST_OR_DICT1[0]), inner_len(LIST_OR_DICT2_TRANSPOSE)
            if _n != _j: dimension_exception("LIST_OR_DICT2_TRANSPOSE", _n, _j)

    del dimension_exception

    return core_hybrid_matmul(LIST_OR_DICT1, LIST_OR_DICT2,
                              LIST_OR_DICT2_TRANSPOSE=LIST_OR_DICT2_TRANSPOSE,
                              return_as=return_as,
                              return_orientation=return_orientation)


def core_hybrid_matmul(LIST_OR_DICT1, LIST_OR_DICT2,
                       LIST_OR_DICT2_TRANSPOSE=None, return_as='SPARSE_DICT',
                       return_orientation='ROW'):
    """Left and right object oriented as []=row, with standard numpy matmul and linear algebra rules. There is
        no protection here to prevent dissimilar sized rows and columns from dotting."""

    if isinstance(LIST_OR_DICT1, dict):
        dict_position = 'LEFT'
        DICT1 = LIST_OR_DICT1
        if LIST_OR_DICT2_TRANSPOSE is None:
            LIST1 = np.array(LIST_OR_DICT2).transpose()
        elif not LIST_OR_DICT2_TRANSPOSE is None:
            LIST1 = np.array(LIST_OR_DICT2_TRANSPOSE)

    elif isinstance(LIST_OR_DICT2, dict):
        dict_position = 'RIGHT'
        LIST1 = np.array(LIST_OR_DICT1)
        if LIST_OR_DICT2_TRANSPOSE is None:
            DICT1 = sparse_transpose(LIST_OR_DICT2)
        elif not LIST_OR_DICT2_TRANSPOSE is None:
            DICT1 = LIST_OR_DICT2_TRANSPOSE

    if len(LIST1.shape) == 1: LIST1 = LIST1.reshape(
        (1, -1))  # DONT RESHAPE OTHERWISE, COULD BE 2-D ARRAY

    if dict_position == 'LEFT' and return_orientation == 'ROW':
        final_outer_len, final_inner_len = outer_len(DICT1), len(LIST1)
    elif dict_position == 'RIGHT' and return_orientation == 'ROW':
        final_outer_len, final_inner_len = len(LIST1), outer_len(DICT1)
    elif dict_position == 'LEFT' and return_orientation == 'COLUMN':
        final_outer_len, final_inner_len = len(LIST1), outer_len(DICT1)
    elif dict_position == 'RIGHT' and return_orientation == 'COLUMN':
        final_outer_len, final_inner_len = outer_len(DICT1), len(LIST1)

    if return_as == 'ARRAY':
        HYBRID_MATMUL = np.zeros((final_outer_len, final_inner_len),
                                 dtype=np.float64)
    elif return_as == 'SPARSE_DICT':
        HYBRID_MATMUL = {int(_): {} for _ in range(final_outer_len)}

    if return_orientation == 'ROW' and dict_position == 'LEFT':
        for outer_idx1 in range(
                final_outer_len if return_orientation == 'ROW' else final_inner_len):
            for outer_idx2 in range(
                    final_inner_len if return_orientation == 'ROW' else final_outer_len):
                dot = core_hybrid_dot(LIST1[outer_idx2],
                                      {0: DICT1[outer_idx1]})
                if dot != 0:
                    HYBRID_MATMUL[int(outer_idx1)][int(outer_idx2)] = dot

    elif return_orientation == 'ROW' and dict_position == 'RIGHT':
        for outer_idx1 in range(
                final_outer_len if return_orientation == 'ROW' else final_inner_len):
            for outer_idx2 in range(
                    final_inner_len if return_orientation == 'ROW' else final_outer_len):
                dot = core_hybrid_dot(LIST1[outer_idx1],
                                      {0: DICT1[outer_idx2]})
                if dot != 0:
                    HYBRID_MATMUL[int(outer_idx1)][int(outer_idx2)] = dot

    elif return_orientation == 'COLUMN' and dict_position == 'LEFT':
        for outer_idx1 in range(
                final_outer_len if return_orientation == 'ROW' else final_inner_len):
            for outer_idx2 in range(
                    final_inner_len if return_orientation == 'ROW' else final_outer_len):
                dot = core_hybrid_dot(LIST1[outer_idx2],
                                      {0: DICT1[outer_idx1]})
                if dot != 0:
                    HYBRID_MATMUL[int(outer_idx2)][int(outer_idx1)] = dot

    elif return_orientation == 'COLUMN' and dict_position == 'RIGHT':
        for outer_idx1 in range(
                final_outer_len if return_orientation == 'ROW' else final_inner_len):
            for outer_idx2 in range(
                    final_inner_len if return_orientation == 'ROW' else final_outer_len):
                dot = core_hybrid_dot(LIST1[outer_idx1],
                                      {0: DICT1[outer_idx2]})
                if dot != 0:
                    HYBRID_MATMUL[int(outer_idx2)][int(outer_idx1)] = dot

    # IF SPARSE_DICT, ENFORCE PLACEHOLDER RULES
    if return_as == 'SPARSE_DICT':
        for outer_key in range(final_outer_len):
            if final_inner_len - 1 not in HYBRID_MATMUL[outer_key]:
                HYBRID_MATMUL[int(outer_key)][int(final_inner_len - 1)] = 0

    return HYBRID_MATMUL








