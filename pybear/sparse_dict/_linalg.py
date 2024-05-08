# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import time
import numpy as np
from pybear.sparse_dict import _validation as val
from pybear.sparse_dict import _linalg_validation as lav
from pybear.sparse_dict._utils import (
                                        outer_len,
                                        inner_len,
                                        inner_len_quick
)



"""
# LINEAR ALGEBRA ##############################################################




** ** **




** ** ** 

core_matmul                     DICT1 and DICT2 enter oriented with sparse inner dicts as rows, with multiplication rules like NumPy.  There is
                                no protection here to prevent dissimilar sized rows from DICT1 dotting with columns from DICT2.
                                Create posn for last entry, so that placeholder rules are enforced (original length of object is retained).
core_symmetric_matmul           DICT1 and DICT2 enter oriented with sparse inner dicts as rows, with multiplication rules like NumPy.
                                For use on things like ATA and AAT to save time.  There is no protection here to prevent dissimilar sized
                                rows from DICT1 dotting with columns from DICT2.  Create posn for last entry, so that placeholder rules are
                                enforced (original length of object is retained).
matmul                          Run matmul with safeguards that assure matrix multiplication rules are followed when running core_matmul().
symmetric_matmul                Run matmul with safeguards that assure matrix multiplication rules are followed when running core_symmetric_matmul().
sparse_ATA                      Calculates ATA on DICT1 using symmetric matrix multiplication.
sparse_AAT                      Calculates AAT on DICT1 using symmetric matrix multiplication.
sparse_matmul_from_lists        Calculates matrix product of two lists and returns as sparse dict.
hybrid_matmul                   Left and right object oriented as []=row, with standard numpy matmul and linear algebra rules, and
                                safeguards that assure matrix multiplication rules are followed when running core_hybrid_matmul().
core_hybrid_matmul              Left and right object oriented as []=row, with standard numpy matmul and linear algebra rules. There is
                                no protection here to prevent dissimilar sized rows and columns from dotting.
                           
** ** ** 
     
core_dot                        Standard dot product. DICT1 and DICT2 enter as single-keyed outer dicts with dict as value.
                                There is no protection here to prevent dissimilar sized DICT1 and DICT2 from dotting.
dot                             Standard dot product.  DICT1 and DICT2 enter as single-keyed outer dicts with dict as value.
                                Run safeguards that assure dot product rules (dimensionality) are followed when running core_dot().
core_gaussian_dot               Gaussian dot product.  [] of DICT1 are dotted with [] from DICT2.  There is no protection here to prevent
                                issimilar sized inner dicts from dotting.
gaussian_dot                    Gaussian dot product.  Run safeguards that assure dot product rules (dimensionality) are followed when running core_dot().
core_hybrid_dot                 Dot product of a single list and one outer sparse dict without any safeguards to ensure single vectors of same length.
# END LINEAR ALGEBRA ##########################################################
"""



def sparse_identity(outer_len: int, inner_len:int, dtype=int) -> dict:

    """Identity matrix as sparse dictionary. Constructs a two dimensional
    identity matrix with positive non-zero dimensions, where elements along the
    diagonal have value 1 and all off-diagonal elements have value zero.
    In other words, a two dimensional matrix where all values are zero except
    the diagonal indices [(0,0), (1,1),.... (n, n)], which are valued 1, where
    n = min(rows, columns).

    Parameters
    ----------
    outer_len:
        int - the number of inner dictionaries; if the inner dictionaries
        hold rows of data, then outer_len is the number of rows.
    inner_len:
        int - the length of the inner dictionaries; if the inner
        dictionaries hold rows of data, then inner_len is the number of columns.
    dtype:
        python or numpy dtype, default = int. The data type of the values in
        the returned sparse dictionary. See Notes.

    Return
    ------
    SPARSE_IDENTITY:
        dict - identity matrix as sparse dictionary with shape
        ('outer_len',  'inner_len') and 'dtype' data type.

    See Also
    --------
    numpy.identity

    Notes
    -----
    There is substantial performance drop-off when using non-python dtypes as
    values in dictionaries. Although use of numpy dtypes is available, the
    recommended dtypes are standard python 'int' and 'float'.

    Examples
    --------
    >>> from pybear.sparse_dict import sparse_identity
    >>> out = sparse_identity(3, 4, dtype=int)
    >>> out
    {0: {0: 1, 3: 0}, 1: {1: 1, 3: 0}, 2: {2: 1, 3: 0}}

    >>> out = sparse_identity(2, 2, dtype=float)
    >>> out
    {0: {0: 1.0, 1: 0.0}, 1: {1: 1.0}}

    """

    err_msg = f" must be an integer greater than 0, and cannot be boolean"
    for name, value in [('outer_len', outer_len), ('inner_len', inner_len)]:
        try:
            float(value)
        except (TypeError, ValueError):
            raise TypeError(f"'{name}'" + err_msg)

        if isinstance(value, bool):
            raise TypeError(f"'{name}'" + err_msg)

        try:
            val._is_int(value)
        except:
            raise TypeError

        if value < 1:
            raise ValueError(f"'{name}'" + err_msg)

    try:
        np.array([], dtype=dtype)
        if dtype is None:
            raise Exception
    except:
        raise ValueError(f"'dtype' must be a valid python or numpy dtype")

    SPARSE_IDENTITY = {}

    for outer_idx in range(outer_len):
        if outer_idx < inner_len - 1:
            SPARSE_IDENTITY[int(outer_idx)] = {int(outer_idx): dtype(1),
                                               int(inner_len - 1): dtype(0)}
        elif outer_idx == inner_len - 1:
            SPARSE_IDENTITY[int(outer_idx)] = {int(outer_idx): dtype(1)}

        elif outer_idx > inner_len - 1:
            SPARSE_IDENTITY[int(outer_idx)] = {int(inner_len - 1): dtype(0)}

    return SPARSE_IDENTITY


# transpose ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **



def core_sparse_transpose(DICT: dict) -> dict:

    """Transpose a sparse dictionary. Safeguards that ensure a properly
    constructed sparse dictionary are not in place for speed.

    Parameters
    ----------
    DICT:
        dict - sparse dictionary to be transposed

    Return
    ------
    DICT:
        dict - transposed sparse dictionary

    See Also
    --------
    numpy.transpose
    pybear.sparse_dict.sparse_transpose
    pybear.sparse_dict.core_sparse_transpose_brute_force
    pybear.sparse_dict.core_sparse_transpose_map

    Examples
    --------
    >>> from pybear.sparse_dict import core_sparse_transpose
    >>> SD = {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}
    >>> SD_T = core_sparse_transpose(SD)
    >>> SD_T
    {0: {0: 1, 1: 3}, 1: {0: 2, 1: 4}}

    """
    
    return core_sparse_transpose_brute_force(DICT)


def sparse_transpose(DICT:dict) -> dict:

    """Transpose a sparse dictionary. Safeguards ensure a properly constructed
    sparse dictionary.

    Parameters
    ----------
    DICT:
        dict - sparse dictionary to be transposed

    Return
    ------
    DICT:
        dict - transposed sparse dictionary

    See Also
    --------
    numpy.transpose
    pybear.sparse_dict.core_sparse_transpose
    pybear.sparse_dict.core_sparse_transpose_brute_force
    pybear.sparse_dict.core_sparse_transpose_map

    Examples
    --------
    >>> from pybear.sparse_dict import sparse_transpose
    >>> SD = {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}
    >>> SD_T = sparse_transpose(SD)
    >>> SD_T
    {0: {0: 1, 1: 3}, 1: {0: 2, 1: 4}}

    """

    val._insufficient_dict_args_1(DICT)
    DICT = val._dict_init(DICT)

    return core_sparse_transpose(DICT)


def core_sparse_transpose_brute_force(DICT: dict) -> dict:

    """Transpose a sparse dictionary by brute force using for loops to read
    values from the original sparse dictionary element-by-element and place
    them into a new transposed array. Safeguards that ensure a properly
    constructed sparse dictionary are not in place for speed.

    Parameters
    ----------
    DICT:
        dict - sparse dictionary to be transposed

    Return
    ------
    DICT1:
        dict - transposed sparse dictionary

    See Also
    --------
    numpy.transpose
    pybear.sparse_dict.core_sparse_transpose
    pybear.sparse_dict.sparse_transpose
    pybear.sparse_dict.core_sparse_transpose_map

    Examples
    --------
    >>> from pybear.sparse_dict import core_sparse_transpose_brute_force
    >>> SD = {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}
    >>> SD_T = sparse_transpose(SD)
    >>> SD_T
    {0: {0: 1, 1: 3}, 1: {0: 2, 1: 4}}

    """

    if DICT == {}:
        return {}

    if len(DICT) == 1 and DICT[list(DICT.keys())[0]] == {}:
        return DICT

    if val._is_sparse_inner(DICT):
        DICT1 = {0: DICT}
    else:
        DICT1 = DICT

    # Inner len becomes outer len, and outer len becomes inner len.
    _lik = max(DICT1)  # last_inner_key
    DICT2 = {int(_): {} for _ in range(inner_len_quick(DICT1))}
    for _outer_key in DICT1:
        for _inner_key in DICT1[_outer_key]:
            DICT2[int(_inner_key)][int(_outer_key)] = DICT1[_outer_key][_inner_key]


    # If at end of each original inner dict (should be a value there due to
    # placeholding from zip() and clean()), skip putting this in TRANSPOSE
    # (otherwise last inner dict in TRANSPOSE would get full of zeros).
    _lok = max(DICT2)  # last_outer_key
    for _inner_key in DICT2[_lok].copy():
        if DICT2[_lok][_inner_key] == 0:
            del DICT2[_lok][_inner_key]

    # If any TRANSPOSE inner dicts do not have a key for outer_len(dict)-1,
    # create one with a zero to maintain placeholding rule.
    for _outer_key in DICT2:
        DICT2[int(_outer_key)][int(_lik)] = DICT2[_outer_key].get(_lik, 0)

    return DICT2


def core_sparse_transpose_map(DICT:dict) -> dict:
    """Transpose a sparse dictionary using map functions to locate values into
    new positions. Safeguards that ensure a properly constructed sparse
    dictionary are not in place for speed.

    Parameters
    ----------
    DICT:
        dict - sparse dictionary to be transposed

    Return
    ------
    DICT:
        dict - transposed sparse dictionary

    See Also
    --------
    numpy.transpose
    pybear.sparse_dict.core_sparse_transpose
    pybear.sparse_dict.sparse_transpose
    pybear.sparse_dict.core_sparse_transpose_brute_force

    Examples
    --------
    >>> from pybear.sparse_dict import core_sparse_transpose_map
    >>> SD = {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}
    >>> SD_T = sparse_transpose(SD)
    >>> SD_T
    {0: {0: 1, 1: 3}, 1: {0: 2, 1: 4}}

    """

    if DICT == {}:
        return {}

    if len(DICT) == 1 and DICT[list(DICT.keys())[0]] == {}:
        return DICT

    if val._is_sparse_inner(DICT):
        DICT1 = {0: DICT}
    else:
        DICT1 = DICT

    old_outer_len = outer_len(DICT1)
    old_inner_len = inner_len_quick(DICT1)


    def placeholder(x):
        NEW_DICT[int(x)][int(old_outer_len - 1)] = \
            NEW_DICT[x].get(old_outer_len - 1, 0)


    def appender(x, outer_key):
        NEW_DICT[int(x)][int(outer_key)] = DICT1[outer_key][x]


    # CREATE TRANSPOSED DICT & FILL WITH {outer_keys:{}}
    NEW_DICT = dict((map(lambda x: (int(x), {}), range(old_inner_len))))

    # POPULATE TRANSPOSED DICT WITH VALUES
    list(map(lambda outer_key: list(
        map(lambda x: appender(x, outer_key), DICT1[outer_key])), DICT1))

    # REMOVE TRANSPOSED PLACEHOLDERS
    list(map(lambda x: NEW_DICT[old_inner_len-1].pop(x),
             [k for k,v in NEW_DICT[old_inner_len-1].items() if v==0]
    ))

    # PLACEHOLDERS
    list(map(lambda x: placeholder(x), NEW_DICT))

    del placeholder, appender

    return NEW_DICT

# END transpose ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **




# matmul ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

def test_new_core_matmul(DICT1, DICT2, DICT2_TRANSPOSE=None, return_as=None):
    """DICT1 and DICT2 enter oriented with sparse inner dicts as rows, with
    multiplication rules like NumPy.  There is no protection here to prevent
    dissimilar sized rows from DICT1 dotting with columns from DICT2. Create
    posn for last entry, so that placeholder rules are enforced (original
    length of object is retained)."""

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
    """DICT1 and DICT2 enter oriented with sparse inner dicts as rows, with
    multiplication rules like NumPy. There is no protection here to prevent
    dissimilar sized rows from DICT1 dotting with columns from DICT2. Create
    posn for last entry, so that placeholder rules are enforced (original
    length of object is retained)."""

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
    """DICT1 and DICT2 enter oriented with sparse inner dicts as rows, with
    multiplication rules like NumPy. For use on things like ATA and AAT to save
    time.  There is no protection here to prevent dissimilar sized rows from
    DICT1 dotting with columns from DICT2.  Create posn for last entry, so that
    placeholder rules are enforced (original length of object is retained)."""

    _display_time = lambda t0: round((time.time() - t0), 2)

    # Transpose DICT2 for ease of multiplication, not for matmul rules.********
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

    lav._symmetric_matmul_check(DICT1, DICT2, DICT1_TRANSPOSE=DICT1_T,
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


# END matmul ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

# dot ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


def core_dot(DICT1:dict, DICT2:dict) -> [int, float]:

    """Inner product of two vectors. There is no protection to prevent
    dissimilar sized DICT1 and DICT2.

    Parameters
    ----------
    DICT1:
        dict - single-keyed outer dictionary or single inner dictionary
    DICT2:
        dict - single-keyed outer dictionary or single inner dictionary

    Return
    ------
    dot:
        [int, float] - inner product of the two vectors

    See Also
    --------
    numpy.dot
    pandas.DataFrame.dot

    Examples
    --------
    >>> from pybear.sparse_dict import core_dot
    >>> dict1 = {0:1, 1:2, 2:0}
    >>> dict2 = {0:2, 2:1}
    >>> dot = core_dot(dict1, dict2)
    >>> dot
    2

    """

    # PIZZA 9-15-22 VERIFIED 5X FASTER THAN CONVERTING TO np.zeros, FILLING,
    # THEN USING np.matmul
    if val._is_sparse_outer(DICT1):
        _DICT1 = DICT1[list(DICT1.keys())[0]]
    else:
        _DICT1 = DICT1

    if val._is_sparse_outer(DICT2):
        _DICT2 = DICT2[list(DICT2.keys())[0]]
    else:
        _DICT2 = DICT2


    dot = 0
    for inner_key1 in _DICT1:
        dot += _DICT1[inner_key1] * _DICT2.get(inner_key1, 0)

    return dot


def dot(DICT1:dict, DICT2:dict) -> [int, float]:
    """Inner product of two vectors. Validation is in place to require sparse
    dictionary vectors of the same length.

    Parameters
    ----------
    DICT1:
        dict - single-keyed outer dictionary or single inner dictionary
    DICT2:
        dict - single-keyed outer dictionary or single inner dictionary

    Return
    ------
    dot:
        [int, float] - inner product of the two vectors

    See Also
    --------
    numpy.dot
    pandas.DataFrame.dot

    Examples
    --------
    >>> from pybear.sparse_dict import dot
    >>> dict1 = {0:1, 1:2, 2:1}
    >>> dict2 = {0:2, 1:1, 2:2}
    >>> out = dot(dict1, dict2)
    >>> out
    6

    """

    val._insufficient_dict_args_2(DICT1, DICT2)
    DICT1 = val._dict_init(DICT1)
    DICT2 = val._dict_init(DICT2)
    lav._dot_size_check(DICT1, DICT2)
    dot = core_dot(DICT1, DICT2)

    return dot


def core_hybrid_dot(LIST1:[list,tuple,np.ndarray], DICT1:dict) -> [int, float]:
    """Inner product of an array-like vector and sparse dictionary vector.
    There is no protection to prevent dissimilar sized LIST1 and DICT2.

    Parameters
    ----------
    LIST1:
        [list, tuple, np.ndarray] - array-like one-dimensional vector
    DICT1:
        dict - single-keyed outer dictionary or single inner dictionary

    Return
    ------
    dot:
        [int, float] - inner product of the two vectors

    See Also
    --------
    pybear.sparse_dict.core_dot
    pybear.sparse_dict.dot

    Examples
    --------
    >>> from pybear.sparse_dict.linalg import core_hybrid_dot
    >>> LIST1 = [1, 2, 3, 4, 5]
    >>> DICT1 = {0: {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}}
    >>> out = hybrid_dot(LIST1, DICT1)
    PIZZA FINISH

    """

    LIST1 = np.array(list(LIST1))
    if len(LIST1.shape) == 1:
        LIST1 = LIST1.reshape((1, -1))

    if val._is_sparse_outer(DICT1):
        _DICT1 = DICT1[list(DICT1.keys())[0]]
    else:
        _DICT1 = DICT1


    dot = 0
    for inner_idx in DICT1[0]:
        # CANT TO USE set.intersection HERE, LIST IS FULL AND NOT INDEXED
        dot += LIST1[0][inner_idx] * DICT1[0][inner_idx]

    return dot


def hybrid_dot(LIST1: [list,tuple,np.ndarray], DICT1:dict) -> [int, float]:
    """Inner product of an array-like vector and sparse dictionary vector.
    Validation is in place to require array-like and sparse dictionary vectors
    of the same length.

    Parameters
    ----------
    LIST1:
        [list, tuple, np.ndarray] - array-like one-dimensional vector
    DICT1:
        dict - single-keyed outer dictionary or single inner dictionary

    Return
    ------
    dot:
        [int, float] - inner product of the two vectors

    See Also
    --------
    pybear.sparse_dict.core_dot
    pybear.sparse_dict.dot

    Examples
    --------
    >>> from pybear.sparse_dict.linalg import hybrid_dot
    >>> LIST1 = [1, 2, 3, 4, 5]
    >>> DICT1 = {0: {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}}
    >>> out = hybrid_dot(LIST1, DICT1)
    PIZZA FINISH

    """

    val._insufficient_list_args_1(LIST1)
    val._insufficient_dict_args_1(DICT1)
    LIST1 = val._list_init(LIST1)
    DICT1 = val._dict_init(DICT1)
    lav._dot_size_check(LIST1, DICT1)


    # if len(LIST1.shape)==1:
    #     LIST1 = LIST1.reshape((1,-1))
    # elif len(LIST1.shape)==2 and len(LIST1) != 1:
    #     raise ValueError(f"")


    dot = core_hybrid_dot(LIST1, DICT1)

    return dot



def core_gaussian_dot(DICT1, DICT2, sigma, return_as='SPARSE_DICT'):
    '''Gaussian dot product.  [] of DICT1 are dotted with [] from DICT2.
        There is no protection here to prevent
        dissimilar sized inner dicts from dotting.'''

    # PIZZA
    from pybear.sparse_dict._transform import unzip_to_ndarray, zip_array

    UNZIP1 = unzip_to_ndarray(DICT1, dtype=np.float64)
    UNZIP2 = unzip_to_ndarray(DICT2, dtype=np.float64)

    final_inner_len = len(UNZIP2)

    GAUSSIAN_DOT = np.zeros((len(UNZIP1), final_inner_len), dtype=np.float64)

    for outer_key1, INNER_DICT1 in enumerate(UNZIP1):
        for outer_key2, INNER_DICT2 in enumerate(UNZIP2):
            GAUSSIAN_DOT[outer_key1][outer_key2] = np.sum((INNER_DICT1 - INNER_DICT2) ** 2)

    del UNZIP1, UNZIP2

    GAUSSIAN_DOT = np.exp(-GAUSSIAN_DOT / (2 * sigma ** 2))

    if return_as == 'SPARSE_DICT':
        GAUSSIAN_DOT = zip_array(GAUSSIAN_DOT, dtype=float)

    return GAUSSIAN_DOT


def gaussian_dot(DICT1, DICT2, sigma, return_as='SPARSE_DICT'):

    DICT1 = val._dict_init(DICT1)
    DICT2 = val._dict_init(DICT2)
    val._inner_len_check(DICT1, DICT2)

    GAUSSIAN_DOT = core_gaussian_dot(DICT1, DICT2, sigma, return_as=return_as)

    return GAUSSIAN_DOT


def core_symmetric_gaussian_dot(DICT1, sigma, return_as='SPARSE_DICT'):
    '''Gaussian dot product for a symmetric result.  [] of DICT1 are dotted with [] from DICT2.  There is no protection
        here to prevent dissimilar sized inner dicts from dotting.'''

    #PIZZA
    from pybear.sparse_dict._transform import unzip_to_ndarray, zip_array

    UNZIP1 = unzip_to_ndarray(DICT1, dtype=np.float64)[0]

    final_inner_len = len(UNZIP1)

    GAUSSIAN_DOT = np.zeros((final_inner_len, final_inner_len),
                            dtype=np.float64)

    for outer_key1 in range(len(UNZIP1)):
        for outer_key2 in range(outer_key1 + 1):  # HAVE TO GET DIAGONAL SO +1
            gaussian_dot = np.sum(
                (UNZIP1[outer_key1] - UNZIP1[outer_key2]) ** 2)
            GAUSSIAN_DOT[outer_key1][outer_key2] = gaussian_dot
            GAUSSIAN_DOT[outer_key2][outer_key1] = gaussian_dot

    del UNZIP1

    GAUSSIAN_DOT = np.exp(-GAUSSIAN_DOT / (2 * sigma ** 2))

    if return_as == 'SPARSE_DICT':
        GAUSSIAN_DOT = zip_array(GAUSSIAN_DOT, dtype=float)

    return GAUSSIAN_DOT





# END dot ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *






if __name__ == '__main__':


    dict1 = {0:1, 1:2, 2:1}
    dict2 = {0:2, 1:1, 2:2}
    out = core_sparse_transpose_brute_force(dict1)
    print(out)





