# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.sparse_dict._utils import (
                                        inner_len,
                                        outer_len,
                                        shape_
)


from pybear.sparse_dict._validation import (
                                            _insufficient_dict_args_2,
                                            _is_sparse_outer,
                                            _is_sparse_inner,
                                            _sparse_dict_check
)

from pybear.sparse_dict._utils import (
                                        core_sparse_equiv,
                                        safe_sparse_equiv
)



def _dot_size_check(DICT1:dict, DICT2:dict):
    """Verify two vectors are sparse dicts that both have unitary outer length
    and equal inner length.
    """

    if _is_sparse_inner(DICT1):
        pass
    else:
        _sparse_dict_check(DICT1)

    if _is_sparse_inner(DICT2):
        pass
    else:
        _sparse_dict_check(DICT2)

    err_msg = f"must be an inner dict, or an outer dict with outer len == 1"

    if _is_sparse_outer(DICT1):
        if outer_len(DICT1) > 1:
            raise ValueError(err_msg)

        _DICT1 = DICT1[list(DICT1.keys())[0]]
    else:
        _DICT1 = DICT1

    if _is_sparse_outer(DICT2):
        if outer_len(DICT2) > 1:
            raise ValueError(err_msg)

        _DICT2 = DICT2[list(DICT2.keys())[0]]
    else:
        _DICT2 = DICT2

    del err_msg


    err_msg = f"requires 2 vectors of equal length"

    if (_DICT1 == {} or _DICT2 == {}) and (_DICT1 != _DICT2):
        raise ValueError(err_msg)

    if inner_len(_DICT1) != inner_len(_DICT2):
        raise ValueError(err_msg)


def _broadcast_check(DICT1:dict, DICT2:dict):
    """Verify the shape of two sparse dicts follow standard matrix
    multiplication rules (m, n) x (j, k), n==j ---> (m, k).
    """

    _insufficient_dict_args_2(DICT1, DICT2)

    if _is_sparse_outer(DICT1) and outer_len(DICT1) == 1:
        _DICT1 = DICT1[list(DICT1.keys())[0]]
    else:
        _DICT1 = DICT1

    if _is_sparse_outer(DICT2) and outer_len(DICT2) == 1:
        _DICT2 = DICT2[list(DICT2.keys())[0]]
    else:
        _DICT2 = DICT2

    err_msg = lambda _, __: (f'DICT1(m x n) and DICT2(j x k) --- '
        f'num inner keys (n) of DICT1 must equal num outer keys (j) of DICT2 --- '
        f'(m, n) x (j, k) --> (m, k) --- {_} != {__}.')

    if (_DICT1 == {} or _DICT2 == {}):
        if _DICT1 != _DICT2:
            raise ValueError(err_msg(shape_(_DICT1), shape_(_DICT2)))
        else:
            return True

    if _is_sparse_inner(_DICT1) and _is_sparse_inner(_DICT2):
        pass

    elif _is_sparse_outer(_DICT1) and _is_sparse_outer(_DICT2):
        _ = shape_(_DICT1)
        __ = shape_(_DICT2)
        if _[1] != __[0]:
            raise ValueError(err_msg(_, __))

    elif _is_sparse_outer(_DICT1) and _is_sparse_inner(_DICT2):
        _ = shape_(_DICT1)
        __ = shape_(_DICT2)
        if _[1] != 1:
            raise ValueError(err_msg(_, __))

    elif _is_sparse_inner(_DICT1) and _is_sparse_outer(_DICT2):
        pass


def _matrix_shape_check(DICT1:dict, DICT2:dict):
    """Verify two sparse dicts have equal shape."""

    _insufficient_dict_args_2(DICT1, DICT2)
    _sparse_dict_check(DICT1)
    _sparse_dict_check(DICT2)

    _shape1, _shape2 = shape_(DICT1), shape_(DICT2)
    if _shape1 != _shape2:
        raise ValueError(f'both sparse dicts must be equally sized. Dict 1 is '
            f'{_shape1} and Dict 2 is {_shape2}')


def _outer_len_check(DICT1:dict, DICT2:dict):
    """Verify two sparse dicts have equal outer length."""

    outer_len1 = outer_len(DICT1)
    outer_len2 = outer_len(DICT2)
    if outer_len1 != outer_len2:
        raise ValueError(
            f'both sparse dicts must have equal outer length.  Dict 1 is '
            f'{outer_len1} and Dict 2 is {outer_len2}')


def _inner_len_check(DICT1:dict, DICT2:dict):
    """Verify two sparse dicts have equal inner length."""

    inner_len1 = inner_len(DICT1)
    inner_len2 = inner_len(DICT2)
    if inner_len1 != inner_len2:
        raise ValueError(f'both sparse dicts must have equal inner length. '
            f'Dict 1 is {inner_len1} and Dict 2 is {inner_len2}')


def _symmetric_matmul_check(DICT1:dict,
                            DICT2:dict,
                            DICT1_TRANSPOSE:dict=None,
                            DICT2_TRANSPOSE:dict=None
    ):
    """Verify two sparse dicts will matrix multiply to a symmetric matrix."""

    from pybear.sparse_dict._linalg import core_sparse_transpose

    _insufficient_dict_args_2(DICT1, DICT2)
    _sparse_dict_check(DICT1)
    _sparse_dict_check(DICT2)
    if DICT1_TRANSPOSE is not None:
        _sparse_dict_check(DICT1_TRANSPOSE)
    if DICT2_TRANSPOSE is not None:
        _sparse_dict_check(DICT2_TRANSPOSE)

    # GET DICT2_T FIRST JUST TO TEST DICT2 IS TRANSPOSE OF DICT1
    if DICT2_TRANSPOSE is not None:
        DICT2_T = DICT2_TRANSPOSE
        # ASSERT DICT2_T REALLY IS T OF DICT2
        if not core_sparse_equiv(DICT2_T, core_sparse_transpose(DICT2)):
            raise ValueError(f'Given DICT2_TRANSPOSE is not the transpose of '
                             f'given DICT2')
    else:
        DICT2_T = core_sparse_transpose(DICT2)

    # TEST DICT2 IS TRANSPOSE OF DICT1
    if safe_sparse_equiv(DICT1, DICT2_T): return True

    # IF DICT2 IS NOT TRANSPOSE OF DICT1, MAKE DICT1_T & PROCEED TO SYMMETRY TESTS
    if DICT1_TRANSPOSE is not None:
        DICT1_T = DICT1_TRANSPOSE
        # ASSERT DICT1_T REALLY IS T OF DICT1
        if not core_sparse_equiv(DICT1_T, core_sparse_transpose(DICT1)):
            raise ValueError(f'Given DICT1_TRANSPOSE is not the transpose of '
                             f'given DICT1')
    else: DICT1_T = core_sparse_transpose(DICT1)

    # TEST BOTH DICT1 AND DICT2 ARE SYMMETRIC
    test2 = safe_sparse_equiv(DICT1, DICT1_T)
    if not test2: return False

    test3 = safe_sparse_equiv(DICT2, DICT2_T)
    if not test3: return False

    return True  # IF GET TO THIS POINT, MUST BE TRUE




