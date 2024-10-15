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
        col_idx: int,
        ignore_float_columns: bool,
        ignore_non_binary_integer_columns: bool
    ) -> tuple[str, dict[DataType: int]]:

    """

    Parallelized fetching of dtype, unqs, and counts from one column of X.

    # 24_03_23_11_28_00 SOMETIMES np.nan IS SHOWING UP MULTIPLE TIMES IN
    # UNIQUES. TROUBLESHOOTING HAS SHOWN THAT THE CONDITION THAT CAUSES
    # THIS IS WHEN dtype(_column_of_X) == object. CONVERT TO np.float64
    # IF POSSIBLE, OTHERWISE GET UNIQUES AS STR

    Parameters
    ----------
    _column_of_X: np.ndarray[DataType] - a single column from X.
    col_idx: int,
    ignore_float_columns: bool,
    ignore_non_binary_integer_columns: bool

    Return
    ------
    -
        tuple[str, dict[DataType: int]] - tuple of dtype and a dictionary.
        dtype can be in ['int', 'float', 'obj'], and the dictionary holds
        uniques as keys and counts as values.

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
        # 24_03_10_15_14_00 IF np.nan IS IN, IT EXISTS AS A str('nan')
        # IN INT AND STR COLUMNS

        UNIQUES = UNIQUES.astype(np.float64)
        # float64 RAISES ValueError ON STR DTYPES, IN THAT CASE MUST BE STR

        # if passed astype, must be numbers from here down

        UNIQUES_NO_NAN = UNIQUES[np.logical_not(nan_mask_numerical(UNIQUES))]

        # determine if is integer
        if np.allclose(UNIQUES_NO_NAN, UNIQUES_NO_NAN.astype(np.int32), atol=1e-6):
            # if is integer and non-binary, return empty if ignoring
            if len(UNIQUES_NO_NAN) > 2 and ignore_non_binary_integer_columns:
                UNQ_CT_DICT = {}
            return 'int', UNQ_CT_DICT
        else:
            # if is not integer, must be float
            if ignore_float_columns:
                UNQ_CT_DICT = {}
            return 'float', UNQ_CT_DICT

    except ValueError:
        # float64 RAISES ValueError ON STR DTYPES, IN THAT CASE MUST BE STR
        try:
            UNIQUES.astype(str)
            return 'obj', UNQ_CT_DICT
        except:
            err_msg = f"Unknown datatype '{UNIQUES.dtype}' in column index {col_idx}"
            raise TypeError(err_msg)
    except:
        err_msg = (f"Removing nans from column index {col_idx} excepted for reason "
                   f"other than ValueError")
        raise Exception(err_msg)











