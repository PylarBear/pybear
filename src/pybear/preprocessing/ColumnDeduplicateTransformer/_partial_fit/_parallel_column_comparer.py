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
    pizza say things

    Parameters
    ----------


    Return
    ------


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



















