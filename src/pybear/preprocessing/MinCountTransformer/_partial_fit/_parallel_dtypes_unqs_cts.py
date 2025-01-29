# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import joblib
import numpy as np
import numpy.typing as npt
from .._type_aliases import DataType
from ....utilities._nan_masking import nan_mask



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
    when dtype(_column_of_X) is object. The problem of multiple nans
    could be fixed by casting object dtype to str, but when the column
    dtype is object, we need to get the uniques in their given dtype,
    not as str. There are much gymnastics done to handle this issue.

    As of 25_01_25 all nan-likes are cast to np.nan in _column_getter().


    Parameters
    ----------
    _column_of_X:
        np.ndarray[DataType] - a single column from X. the 'data'
        attribute of the single column if X was passed as a scipy sparse
        matrix/array.
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


    if _column_of_X.dtype == object:

        # remember the nan notes in the docstring, that when object multiple
        # nans show up in uniques. changing the dtype to str works for
        # getting only one nan, but this changes any numbers in the obj
        # column to string also, which persists even when changing the dtype
        # of the column back to obj, which then populates UNQ_CT_DICT with
        # str(num). this opens a huge can of worms for making row masks. so
        # we need to get the uniques in the original dtype, then if object,
        # remove any extra nans. DO NOT MUTATE _column_of_X DTYPES!

        # but, there is another huge problem here, numpy.unique cant do
        # the mixed dtypes that might be in an 'object' column.
        # TypeError: '<' not supported between instances of 'float' and 'str'
        # we need to separate out num from str.

        NUM_LIKE = []
        STR_LIKE = []
        for _value in _column_of_X:
            try:
                float(_value)
                NUM_LIKE.append(_value)
            except:
                STR_LIKE.append(_value)

        NUM_LIKE_UNQ_CT_DICT = \
            dict((zip(*np.unique(NUM_LIKE, return_counts=True))))
        STR_LIKE_UNQ_CT_DICT = \
            dict((zip(*np.unique(STR_LIKE, return_counts=True))))

        del NUM_LIKE, STR_LIKE

        UNQ_CT_DICT = NUM_LIKE_UNQ_CT_DICT | STR_LIKE_UNQ_CT_DICT
        del NUM_LIKE_UNQ_CT_DICT, STR_LIKE_UNQ_CT_DICT

        UNQS = np.fromiter(UNQ_CT_DICT.keys(), dtype=object)
        CTS = np.fromiter(map(int, UNQ_CT_DICT.values()), dtype=int)
        # if more than 1 nan, chop out all of them after the first, but
        # dont forget to put the total of all the nans on the one kept!
        NANS = nan_mask(UNQS).astype(bool)
        if np.sum(NANS) > 1:
            NAN_IDXS = np.arange(len(UNQS))[NANS]
            CHOP_NAN_IDXS = NAN_IDXS[1:]
            CHOP_COUNTS = int(np.sum(CTS[CHOP_NAN_IDXS]))
            CTS[NAN_IDXS[0]] = (CTS[NAN_IDXS[0]] + CHOP_COUNTS)
            del NAN_IDXS, CHOP_COUNTS
            CHOP_NAN_BOOL = np.ones(len(UNQS)).astype(bool)
            CHOP_NAN_BOOL[CHOP_NAN_IDXS] = False
            del CHOP_NAN_IDXS
            NEW_UNQS = UNQS[CHOP_NAN_BOOL]
            NEW_CTS = CTS[CHOP_NAN_BOOL]
            UNQ_CT_DICT = dict((zip(NEW_UNQS, map(int, NEW_CTS))))
            del NEW_UNQS, NEW_CTS
        del UNQS, CTS, NANS
    else:
        UNQ_CT_DICT = dict((zip(*np.unique(_column_of_X, return_counts=True))))


    UNQ_CT_DICT = dict((zip(
        np.fromiter(UNQ_CT_DICT.keys(), dtype=_column_of_X.dtype),
        map(int, UNQ_CT_DICT.values())
    )))

    UNIQUES = np.fromiter(UNQ_CT_DICT.keys(), dtype=_column_of_X.dtype)

    UNIQUES_NO_NAN = UNIQUES[np.logical_not(nan_mask(UNIQUES))]

    del UNIQUES


    # float64 RAISES ValueError ON STR DTYPES, IN THAT CASE MUST BE STR
    # if accepted astype, must be numbers from here down
    try:
        _column_of_X.astype(np.float64)
    except:
        return 'obj', UNQ_CT_DICT



    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    # hidden zeros only apply to scipy, and scipy can only be numeric
    _has_hidden_zeros = (_n_rows - len(_column_of_X))

    if _has_hidden_zeros:
        # then is .data attribute from scipy, get the zeros
        ZERO_ADDON_DICT = {0: _n_rows - len(_column_of_X)}
    else:
        ZERO_ADDON_DICT = {}
    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # determine if is integer
    if np.allclose(
        UNIQUES_NO_NAN.astype(np.float64),
        UNIQUES_NO_NAN.astype(np.float64).astype(np.int32),
        atol=1e-6
    ):
        if (len(UNIQUES_NO_NAN) + _has_hidden_zeros) <= 2:
            return 'bin_int', ZERO_ADDON_DICT | UNQ_CT_DICT
        else:
            return 'int', ZERO_ADDON_DICT | UNQ_CT_DICT
    else:
        return 'float', ZERO_ADDON_DICT | UNQ_CT_DICT











