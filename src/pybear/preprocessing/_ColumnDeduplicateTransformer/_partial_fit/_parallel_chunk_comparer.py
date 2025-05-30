# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Any
import numpy.typing as npt

import numbers

import numpy as np
import joblib

from ....utilities._nan_masking import nan_mask



@joblib.wrap_non_picklable_objects
def _parallel_chunk_comparer(
    _chunk1: npt.NDArray[Any],
    _chunk1_X_indices: tuple[int],
    _chunk2: npt.NDArray[Any],
    _chunk2_X_indices: tuple[int],
    _rtol: numbers.Real,
    _atol: numbers.Real,
    _equal_nan: bool
) -> list[bool]:

    """
    Compare the columns in chunk1 against the columns in chunk2 for
    equality, subject to rtol, atol, and equal_nan. chunk1 and chunk2
    must be ndarray.


    Parameters
    ----------
    _chunk1:
        npt.NDArray[Any] - a chunk from X to compare against another
        chunk of columns from X for equality.
    _chunk2:
        npt.NDArray[Any] - the other chunk of columns from X.
    _rtol:
        numbers.Real - The relative difference tolerance for equality.
        Must be a non-boolean, non-negative, real number.
        See numpy.allclose.
    _atol:
        numbers.Real - The absolute difference tolerance for equality.
        Must be a non-boolean, non-negative, real number.
        See numpy.allclose.
    _equal_nan:
        bool - When comparing pairs of columns row by row:
        If equal_nan is True, exclude from comparison any rows where one
        or both of the values is/are nan. If one value is nan, this
        essentially assumes that the nan value would otherwise be the
        same as its non-nan counterpart. When both are nan, this
        considers the nans as equal (contrary to the default numpy
        handling of nan, where numpy.nan != numpy.nan) and will not in
        and of itself cause a pair of columns to be considered unequal.
        If equal_nan is False and either one or both of the values in
        the compared pair of values is/are nan, consider the pair to be
        not equivalent, thus making the column pair not equal. This is
        in line with the normal numpy handling of nan values.


    Return
    ------
    _pairs:
        list[tuple[int, int]] - The pairs of columns that are equal
        between chunk1 and chunk2.

    """


    # validation ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
    assert isinstance(_chunk1, np.ndarray)
    assert isinstance(_chunk2, np.ndarray)

    assert isinstance(_chunk1_X_indices, tuple)
    assert all(map(isinstance, _chunk1_X_indices, (int for i in _chunk1_X_indices)))
    assert len(_chunk1_X_indices) == _chunk1.shape[1]

    assert isinstance(_chunk2_X_indices, tuple)
    assert all(map(isinstance, _chunk2_X_indices, (int for i in _chunk2_X_indices)))
    assert len(_chunk2_X_indices) == _chunk2.shape[1]

    try:
        float(_rtol)
        assert _rtol >= 0
        float(_atol)
        assert _atol >= 0
    except:
        raise ValueError(
            f"'rtol' and 'atol' must be real, non-negative numbers"
        )
    # END validation ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **


    _pairs = []
    for _chunk1_idx, _X1_idx in enumerate(_chunk1_X_indices):

        _column1 = _chunk1[:, _chunk1_idx].ravel()

        try:
            # float64 RAISES ValueError ON STR DTYPES, IN THAT CASE MUST BE STR
            _column1.astype(np.float64)
            _column1_is_num = True
        except:
            _column1_is_num = False


        for _chunk2_idx, _X2_idx in enumerate(_chunk2_X_indices):

            if _X2_idx <= _X1_idx:
                # do not double count
                continue

            _column2 = _chunk2[:, _chunk2_idx].ravel()

            try:
                # float64 RAISES ValueError ON STR DTYPES, IN THAT CASE MUST BE STR
                _column2.astype(np.float64)
                _column2_is_num = True
            except:
                _column2_is_num = False


            MASK1 = nan_mask(_column1)
            MASK2 = nan_mask(_column2)
            NOT_NAN_MASK = np.logical_not((MASK1 + MASK2).astype(bool))
            del MASK1, MASK2

            if _column1_is_num and _column2_is_num:

                if _equal_nan:
                    _match = np.allclose(
                        _column1[NOT_NAN_MASK].astype(np.float64),
                        _column2[NOT_NAN_MASK].astype(np.float64),
                        rtol=_rtol,
                        atol=_atol
                    )

                elif not _equal_nan:
                    _match = np.allclose(
                        _column1.astype(np.float64),
                        _column2.astype(np.float64),
                        rtol=_rtol,
                        atol=_atol
                    )

            elif not _column1_is_num and not _column2_is_num:

                if _equal_nan:
                    _match = np.array_equal(
                        _column1[NOT_NAN_MASK].astype(object),
                        _column2[NOT_NAN_MASK].astype(object)
                    )

                elif not _equal_nan:
                    if not all(NOT_NAN_MASK):
                        _match = False
                    else:
                        _match = np.array_equal(
                            _column1.astype(object),
                            _column2.astype(object)
                        )

            else:
                # if one column is num and another column is not num, cannot be
                # equal
                _match = False

            if _match:
                _pairs.append((_X1_idx, _X2_idx))


            del _column2, _column2_is_num, NOT_NAN_MASK


    return _pairs



