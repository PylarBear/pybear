# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import numpy.typing as npt

import numpy as np
import joblib
from numbers import Real

from ....utilities._nan_masking import nan_mask



@joblib.wrap_non_picklable_objects
def _parallel_column_comparer(
    _column1: npt.NDArray[any],
    _column2: npt.NDArray[any],
    _rtol: Real,
    _atol: Real,
    _equal_nan: bool
) -> bool:

    """
    Compare two columns for equality, subject to rtol, atol, and
    equal_nan. The columns here must have originated from a numpy array
    or pandas dataframe, not a scipy sparse matrix/array. For the scipy
    column comparer, see _parallel_ss_comparer.


    Parameters
    ----------
    _column1:
        npt.NDArray[any] - the first column of a pair to compare for
        equality.
    _column2:
        npt.NDArray[any] - the second column of a pair to compare for
        equality.
    _rtol:
        numbers.Real - The relative difference tolerance for equality.
        Must be a non-boolean, non-negative, real number. See
        numpy.allclose.
    _atol:
        numbers.Real - The absolute difference tolerance for equality.
        Must be a non-boolean, non-negative, real number. See
        numpy.allclose.
    _equal_nan:
        bool - When comparing pairs of columns row by row:
        If equal_nan is True, exclude from comparison any rows where one
        or both of the values is/are nan. If one value is nan, this
        essentially assumes that the nan value would otherwise be the
        same as its non-nan counterpart. When both are nan, this
        considers the nans as equal (contrary to the default numpy
        handling of nan, where numpy.nan != numpy.nan) and will not in
        and of itself cause a pair of columns to be marked as unequal.
        If equal_nan is False and either one or both of the values in
        the compared pair of values is/are nan, consider the pair to be
        not equivalent, thus making the column pair not equal. This is
        in line with the normal numpy handling of nan values.


    Return
    ------
    -
        bool: True, the columns are equal, False, the columns are unequal.

    """

    # pizza
    # assert isinstance(_column1, np.ndarray)
    # assert isinstance(_column2, np.ndarray)


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
                return np.array_equal(
                    _column1.astype(object),
                    _column2.astype(object)
                )

    else:
        # if one column is num and another column is not num, cannot be
        # equal
        return False



















