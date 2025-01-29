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
def _parallel_dtypes_unqs_cts(
    _column_of_X: npt.NDArray[DataType],
    _n_rows: int,
    _col_idx: int
) -> tuple[str, dict[DataType, int]]:


    """
    Parallelized collection of dtype, uniques, and frequencies from one
    column of X.

    *** VERY IMPORTANT *** when ss columns are extracted, only the data
    attribute is sent here. Need to infer the number of zeros in the
    column. The difference _n_rows - len(_column_of_X) is the number of
    zeros in the column.

    Sometimes np.nan is showing up multiple times in uniques.
    Troubleshooting has shown that the condition that causes this is
    when dtype(_column_of_X) is object. Convert to np.float64 if
    possible, otherwise get uniques as str. As of 25_01_25 all nan-likes
    are cast to np.nan in _column_getter().


    Parameters
    ----------
    _column_of_X:
        np.ndarray[DataType] - a single column from X.
    _n_rows:
        int - the number of samples in X.
    _col_idx:
        int - the column index occupied by _column_of_X in the data.
        this is for error reporting purposes only.


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

        _has_hidden_zeros = (_n_rows - len(_column_of_X))

        if _has_hidden_zeros:
            # then is .data attribute from scipy, get the zeros
            ZERO_ADDON_DICT = {0: _n_rows - len(_column_of_X)}
        else:
            ZERO_ADDON_DICT = {}

        # determine if is integer
        if np.allclose(
            UNIQUES_NO_NAN,
            UNIQUES_NO_NAN.astype(np.int32),
            atol=1e-6
        ):
            if (len(UNIQUES_NO_NAN) + _has_hidden_zeros) <= 2:
                return 'bin_int', ZERO_ADDON_DICT | UNQ_CT_DICT
            else:
                return 'int', ZERO_ADDON_DICT | UNQ_CT_DICT
        else:
            return 'float', ZERO_ADDON_DICT | UNQ_CT_DICT

    except ValueError:
        # float64 RAISES ValueError ON STR DTYPES, IN THAT CASE MUST BE STR
        # IF np.nan IS IN, IT EXISTS AS A str('nan')
        # scipy sparse cant get in here, cant take obj dtype
        try:
            UNIQUES.astype(str)
            return 'obj', UNQ_CT_DICT
        except:
            raise TypeError(
                f"Unknown datatype '{UNIQUES.dtype}' in column index {_col_idx}"
            )
    except:
        raise Exception(
            f"Removing nans from column index {_col_idx} excepted for reason "
            f"other than ValueError"
        )











