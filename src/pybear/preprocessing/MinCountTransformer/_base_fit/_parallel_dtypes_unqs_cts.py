# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import joblib
import numpy as np
import numpy.typing as npt
from .._type_aliases import DataType
from ....utilities._nan_masking import nan_mask_numerical



@joblib.wrap_non_picklable_objects
def _dtype_unqs_cts_processing(
        _column_of_X: npt.NDArray[DataType],
        col_idx: int
    ) -> tuple[str, dict[DataType, int]]:

    """

    Parallelized fetching of dtype, unqs, and counts from one column of X.

    # 24_03_23 SOMETIMES np.nan IS SHOWING UP MULTIPLE TIMES IN
    # UNIQUES. TROUBLESHOOTING HAS SHOWN THAT THE CONDITION THAT CAUSES
    # THIS IS WHEN dtype(_column_of_X) == object. CONVERT TO np.float64
    # IF POSSIBLE, OTHERWISE GET UNIQUES AS STR


    Parameters
    ----------
    _column_of_X:
        np.ndarray[DataType] - a single column from X.
    col_idx: int


    Return
    ------
    -
        tuple[str, dict[DataType, int]] - tuple of dtype and a dictionary.
        dtype can be in ['bin_int', 'int', 'float', 'obj'], and the
        dictionary holds uniques as keys and counts as values.

    """


    _column_orig_dtype = _column_of_X.dtype
    try:
        UNQ_CT_DICT = dict((
            zip(*np.unique(_column_of_X.astype(np.float64),return_counts=True))
        ))
    except:
        UNQ_CT_DICT = dict((
            zip(*np.unique(_column_of_X.astype(str), return_counts=True))
        ))

    UNQ_CT_DICT = dict((zip(
            np.fromiter(UNQ_CT_DICT.keys(), dtype=_column_orig_dtype),
            map(int, UNQ_CT_DICT.values())
    )))

    UNIQUES = np.fromiter(UNQ_CT_DICT.keys(), dtype=_column_orig_dtype)
    del _column_orig_dtype

    try:

        UNIQUES = UNIQUES.astype(np.float64)
        # float64 RAISES ValueError ON STR DTYPES, IN THAT CASE MUST BE STR

        # if accepted astype, must be numbers from here down
        UNIQUES_NO_NAN = UNIQUES[np.logical_not(nan_mask_numerical(UNIQUES))]

        # determine if is integer
        if np.allclose(
            UNIQUES_NO_NAN,
            UNIQUES_NO_NAN.astype(np.int32),
            atol=1e-6
        ):
            if len(UNIQUES_NO_NAN) <= 2:
                return 'bin_int', UNQ_CT_DICT
            else:
                return 'int', UNQ_CT_DICT
        else:
            return 'float', UNQ_CT_DICT

    except ValueError:
        # float64 RAISES ValueError ON STR DTYPES, IN THAT CASE MUST BE STR
        # IF np.nan IS IN, IT EXISTS AS A str('nan')
        try:
            UNIQUES.astype(str)
            return 'obj', UNQ_CT_DICT
        except:
            raise TypeError(
                f"Unknown datatype '{UNIQUES.dtype}' in column index {col_idx}"
            )
    except:
        raise Exception(
            f"Removing nans from column index {col_idx} excepted for reason "
            f"other than ValueError"
        )











