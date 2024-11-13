# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import numpy.typing as npt

import numpy as np
import joblib

from ....utilities._nan_masking import nan_mask



@joblib.wrap_non_picklable_objects
def _parallel_column_comparer(
    _column1: npt.NDArray[any],
    _column2: npt.NDArray[any],
    _rtol: float,
    _atol: float,
    _equal_nan: bool
) -> bool:

    """
    Compare two columns for equality, subject to :param: rtol, :param:
    atol, and :param: equal_nan.


    Parameters
    ----------
    _column1:
        npt.NDArray[any] - the first column of a pair to compare for
        equality.
    _column2:
        npt.NDArray[any] - the second column of a pair to compare for
        equality.
    _rtol:
        float, default = 1e-5 - The relative difference tolerance for
            equality. See numpy.allclose.
    _atol:
        float, default = 1e-8 - The absolute tolerance parameter for .
            equality. See numpy.allclose.
    _equal_nan:
        bool, default = False - When comparing pairs of columns row by
        row:
        If equal_nan is True, exclude from comparison any rows where one
        or both of the values is/are nan. If one value is nan, this
        essentially assumes that the nan value would otherwise be the
        same as its non-nan counterpart. When both are nan, this
        considers the nans as equal (contrary to the default numpy
        handling of nan, where np.nan != np.nan) and will not in and of
        itself cause a pair of columns to be marked as unequal.
        If equal_nan is False and either one or both of the values in
        the compared pair of values is/are nan, consider the pair to be
        not equivalent, thus making the column pair not equal. This is
        in line with the normal numpy handling of nan values.


    Return
    ------
    -
        bool: True, the columns are equal, False, the columns are uneqaul.

    """


    try:
        # float64 RAISES ValueError ON STR DTYPES, IN THAT CASE MUST BE STR
        _column1.astype(np.float64)
        _column1_is_num = True
    except:
        _column1_is_num = False


    try:
        # float64 RAISES ValueError ON STR DTYPES, IN THAT CASE MUST BE STR
        _column2.astype(np.float64)
        _column2_is_num = True
    except:
        _column2_is_num = False


    MASK1 = nan_mask(_column1)
    MASK2 = nan_mask(_column2)

    # if using scipy sparse, the "column" being compared is an hstack of
    # the indices and values of the dense in that column. It is very
    # possible that the two vectors being compared will have different
    # length, and the numpy vectorization used below will not broadcast.
    # Compare the lengths of the vectors here and if different, short
    # circuit out.
    # pizza come back and finish this
    # if not _equal_nan:
    #     if len(_column1) != len(_column2):
    #         return False
    #     # else column lengths are equal so proceed to tests below
    # elif _equal_nan:
    #     if len(_column1[NOT_NAN_MASK1]) != len(_column2[NOT_NAN_MASK2]):
    #         return False
    #     # else column lengths are equal so proceed to tests below



    NOT_NAN_MASK = np.logical_not((MASK1 + MASK2).astype(bool))
    del MASK1, MASK2


    if _column1_is_num and _column2_is_num:

        if _equal_nan:

            return np.allclose(
                _column1[NOT_NAN_MASK].astype(np.float64),
                _column2[NOT_NAN_MASK].astype(np.float64),
                rtol=_rtol,
                atol=_atol
            )

        elif not _equal_nan:
            return np.allclose(
                _column1.astype(np.float64),
                _column2.astype(np.float64),
                rtol=_rtol,
                atol=_atol
            )

    elif not _column1_is_num and not _column2_is_num:

        if _equal_nan:

            return np.array_equal(
                _column1[NOT_NAN_MASK].astype(object),
                _column2[NOT_NAN_MASK].astype(object)
            )

        elif not _equal_nan:
            if not all(NOT_NAN_MASK):
                return False
            else:
                return np.array_equal(_column1.astype(object), _column2.astype(object))

    else:
        # if one column is num and another column is not num, cannot be
        # equal
        return False



















