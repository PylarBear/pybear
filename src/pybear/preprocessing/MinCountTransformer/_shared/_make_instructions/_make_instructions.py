# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from typing_extensions import Union
from copy import deepcopy
import numpy as np

from ..._type_aliases import (
    TotalCountsByColumnType,
    InstructionsType,
    OriginalDtypesDtype,
)

from ._make_instructions_validation import _make_instructions_validation

from ._one_unique import _one_unique
from ._two_uniques_hab import _two_uniques_hab
from ._two_uniques_not_hab import _two_uniques_not_hab
from ._three_or_more_uniques_hab import _three_or_more_uniques_hab
from ._three_or_more_uniques_not_hab import _three_or_more_uniques_not_hab
from .....utilities._nan_masking import nan_mask_numerical


# _make_instructions()
#   CALLED in MCT BY get_support(), test_threshold(), AND transform()
# get_support()
#   CALLED in MCT BY get_feature_names(), inverse_transform() AND transform()



def _make_instructions(
        _count_threshold: int,
        _ignore_float_columns: bool,
        _ignore_non_binary_integer_columns: bool,
        _ignore_columns: np.ndarray[int],
        _ignore_nan: bool,
        _handle_as_bool: np.ndarray[int],
        _delete_axis_0: bool,
        _original_dtypes: OriginalDtypesDtype,
        _n_features_in: int,
        _total_counts_by_column: TotalCountsByColumnType,
        _threshold: Union[int, None] = None
    ) -> InstructionsType:

    """
    Convert compiled uniques and frequencies into instructions for
    transforming data based on given parameters.

    _delete_instr is a dictionary that is keyed by column index and the
    values are lists. Within the lists is information about operations
    to perform with respect to values in the column. The following
    items may be in the list:
    -- 'INACTIVE' - ignore the column and carry it through for all other
        operations
    -- individual values (in raw datatype format, not converted to
        string) indicates to delete the rows on axis 0 that contain that
        value in that column, including 'nan' or np.nan values
    -- 'DELETE COLUMN' - perform any individual row deletions that need
        to take place while the column is still in the data, then delete
        the column from the data.

    _total_counts_by_column is a dictionary that is keyed by column index
    and the values are dictionaries. Each inner dictionary is keyed by
    the uniques in that column and the values are the respective counts
    in that column. 'nan' are documented in these dictionaries, which
    complicates assessing if columns have 1, 2, or 3+ unique values. If
    _ignore_nan, nans and their counts are removed from the dictionaries
    and set aside while the remaining non-nan values are processed by
    _make_instructions.

    Instructions are made based on the counts found in all the training
    data seen up to the point of calling _make_instructions. Counts can
    be accreted incrementally, as with partial_fit(), and then finally
    _make_instructions is run to create instructions based on the total
    counts. This allows for use of a dask Incremental wrapper to get
    counts for bigger-than-memory data. Because the instructions are
    agnostic to the origins of the data they were created from, the
    instructions can be applied to any data that matches the schema of
    the training data. This allows for the use of an incremental post-fit
    like dask ParallelPostFit. This also allows for transformation of
    unseen data.

    This module makes delete instructions based on the uniques and counts
    as found as it iterates over the columns in _total_counts_by_column.

    A) if col_idx is inactive, skip.
    column is 'INACTIVE' if:
       - col_idx in _ignore_columns:
       - _total_counts_by_column[col_idx] is empty
       - _ignore_float_columns and is float column
       - _ignore_non_binary_integer_columns, is 'int', and num unqs >= 3

    B) MANAGE nan
        Create holder objects to hold the value and the count.
        1) Get nan information if nan is in _total_counts_by_column[col_idx]
        - if ignore_nan, holder objects hold False
        - if not ignoring nan:
          -- 'nan' not in column, holder objects hold False
          -- 'nan' in column, holder objects hold the nan value and ct
        2) Temporarily remove nan and its count from _total_counts_by_column,
            if ignoring or not

    C) Assess the remaining values and counts and create instructions.
    Now that nan is out of _total_counts_by_column, look at the number of
    items in uniques and direct based on if is 1, 2, or 3+, and if
    handling as boolean.

    There are 5 modules called by this module. All cases where there is
    only one unique in a feature are handled the same, by one module.
    Otherwise, the algorithms for handling as boolean and not handling
    as boolean are separate for the cases of 2 uniques or 3+ uniques.

    See the individual modules for detailed explanation of the logic.


    Parameters
    ----------
    _count_threshold:
        int - The threshold that determines whether a value is removed
        from the data (frequency is below threshold) or retained (frequency
        is greater than or equal to threshold.)

    _ignore_float_columns:
        bool - If True, values and frequencies within float features are
        ignored and the feature is retained through transform. If False,
        the feature is handled as if it is categorical and unique values
        are subject to count threshold rules and possible removal.

    _ignore_non_binary_integer_columns:
        bool - If True, values and frequencies within non-binary integer
        features are ignored and the feature is retained through transform.
        If False, the feature is handled as if it is categorical and
        unique values are subject to count threshold rules and possible
        removal.

    _ignore_columns:
        np.ndarray[int] - A one-dimensional vector of integer index
        positions. Excludes indicated features from the thresholding
        operation.

    _ignore_nan:
        bool - If True, nan is ignored in all features and passes through
        the transform operation; it can only be removed collaterally by
        removal of examples for causes dictated by other features. If
        False, frequency for both numpy.nan and 'nan' as string are
        calculated and assessed against count_threshold.

    _handle_as_bool:
        np.ndarray[int] - A one-dimensional vector of integer index
        positions. For the indicated features, non-zero values within
        the feature are treated as if they are the same value.

    _delete_axis_0:
        bool - Only applies to features indicated in :param:
        handle_as_bool or binary integer features such as those generated
        by OneHotEncoder. Under normal operation of MinCountTransformer
        for datatypes other than binary integer, when the frequency of
        one of the values in the feature is below :param: count_threshold,
        the respective examples are removed. However, binary integers
        do not behave like this in that the rows of deficient frequency
        are not removed, only the entire column is removed. :param:
        delete_axis_0 overrides this behavior. When :param: delete_axis_0
        is False under the above conditions, MinCountTransformer does the
        default behavior for binary integers, the feature is removed
        without deleting examples, preserving the data in the other
        features. If True, however, the default behavior for other
        datatypes is used and examples associated with the minority value
        are removed and the feature is then also removed for having only
        one value.

    _original_dtypes:
        np.ndarray[str] - The original datatypes for each column in the
        dataset as determined by MinCountTransformer. Values can be
        'bin_int', 'int', 'float', or 'obj'.

    _n_features_in:
        int - the number of features (columns) in the dataset.

    _total_counts_by_column:
        dict[int, dict[DataType, int]] - a zero-indexed dictionary that
        holds dictionaries containing the counts of the uniques in each
        column.

    _threshold:
        int - The threshold that determines whether a value is removed
        from the data (frequency is below threshold) or retained (frequency
        is greater than or equal to threshold.) If not provided,
        _count_threshold is used; if provided, overwrites _count_threshold
        to allow for tests of thresholds different than that defined by
        _count_threshold in the instance.



    Returns
    ----------
    -
        _delete_instr: dict[int, Union[str, DataType]]



    """


    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *



    # _validation
    _count_threshold, _ignore_float_columns, _ignore_non_binary_integer_columns, \
    _ignore_columns, _ignore_nan, _handle_as_bool, _delete_axis_0, \
    _original_dtypes, _n_features_in, _total_counts_by_column, _threshold = \
        _make_instructions_validation(
            _count_threshold,
            _ignore_float_columns,
            _ignore_non_binary_integer_columns,
            _ignore_columns,
            _ignore_nan,
            _handle_as_bool,
            _delete_axis_0,
            _original_dtypes,
            _n_features_in,
            _total_counts_by_column,
            _threshold
        )



    # find inactive columns and populate _delete_instr
    _delete_instr = {}
    for col_idx in range(_n_features_in):
        _delete_instr[col_idx] = []
        if col_idx in _ignore_columns:
            _delete_instr[col_idx].append('INACTIVE')
        elif _total_counts_by_column[col_idx] == {}:
            # {} is acceptable here, when ignore <any column, float columns
            # or non-bin-int columns>, _dtype_unqs_cts_processing returns
            # UNQ_CT_DICT as {}
            _delete_instr[col_idx].append('INACTIVE')
        elif (_ignore_float_columns and _original_dtypes[col_idx] == 'float'):
            _delete_instr[col_idx].append('INACTIVE')
        elif _ignore_non_binary_integer_columns and \
                _original_dtypes[col_idx] == 'int':
            UNQS = np.fromiter(
                _total_counts_by_column[col_idx].keys(),
                dtype=np.float64
            )
            if len(UNQS[np.logical_not(nan_mask_numerical(UNQS))]) >= 3:
                _delete_instr[col_idx].append('INACTIVE')
            del UNQS



    for col_idx, COLUMN_UNQ_CT_DICT in deepcopy(_total_counts_by_column).items():
        # _total_counts_by_column GIVES A DICT OF UNQ & CTS FOR COLUMN;
        # IF _ignore_nan, nans & THEIR CTS ARE TAKEN OUT BELOW. IF nan IS
        # IN, THIS COMPLICATES ASSESSMENT OF COLUMN HAS 1 VALUE, IS BINARY, ETC.

        if _delete_instr[col_idx] == ['INACTIVE']:
            continue

        # vvv MANAGE nan ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # GET OUT nan INFORMATION IF nan IS IN
        if _ignore_nan:
            _nan_key = False
            _nan_ct = False
        else:
            _nan_dict = {k: v for k, v in COLUMN_UNQ_CT_DICT.items()
                         if str(k).lower() == 'nan'}
            if len(_nan_dict) == 0:
                _nan_key = False
                _nan_ct = False
            elif len(_nan_dict) == 1:
                _nan_key = list(_nan_dict.keys())[0]
                _nan_ct = list(_nan_dict.values())[0]
            else:
                raise AssertionError(f">=2 nans found in COLUMN_UNQ_CT_DICT")

            del _nan_dict

        # TEMPORARILY REMOVE nan FROM COLUMN_UNQ_CT_DICT, WHETHER IGNORING OR NOT
        COLUMN_UNQ_CT_DICT = {unq: ct for unq, ct in COLUMN_UNQ_CT_DICT.items()
                              if str(unq).lower() != 'nan'}
        # ^^^ END MANAGE nan ** * ** * ** * ** * ** * ** * ** * ** * ** * **

        if len(COLUMN_UNQ_CT_DICT) == 1:  # SAME VALUE IN THE WHOLE COLUMN

            _delete_instr[col_idx] = _one_unique(
                _delete_instr[col_idx],
                _threshold,
                _nan_key,
                _nan_ct,
                COLUMN_UNQ_CT_DICT
            )

        elif len(COLUMN_UNQ_CT_DICT) == 2:  # BINARY, ANY DTYPE

            if _original_dtypes[col_idx] == 'bin_int' or col_idx in _handle_as_bool:
                _delete_instr[col_idx] = _two_uniques_hab(
                    _delete_instr[col_idx],
                    _threshold,
                    _nan_key,
                    _nan_ct,
                    COLUMN_UNQ_CT_DICT,
                    _delete_axis_0
                )
            else:
                _delete_instr[col_idx] = _two_uniques_not_hab(
                    _delete_instr[col_idx],
                    _threshold,
                    _nan_key,
                    _nan_ct,
                    COLUMN_UNQ_CT_DICT
                )

        else:  # 3+ UNIQUES NOT INCLUDING nan

            if col_idx == 3:
                print(f'pizza goes into the right oven!')

            if col_idx in _handle_as_bool:

                if _original_dtypes[col_idx] == 'obj':
                    raise ValueError(f"handle_as_bool on str column")

                _delete_instr[col_idx] = _three_or_more_uniques_hab(
                    _delete_instr[col_idx],
                    _threshold,
                    _nan_key,
                    _nan_ct,
                    COLUMN_UNQ_CT_DICT,
                    _delete_axis_0
                )
            else:
                _delete_instr[col_idx] = _three_or_more_uniques_not_hab(
                    _delete_instr[col_idx],
                    _threshold,
                    _nan_key,
                    _nan_ct,
                    COLUMN_UNQ_CT_DICT
                )



    del _threshold, col_idx, COLUMN_UNQ_CT_DICT
    try:
        _nan_key, _nan_ct, _nan_dict
    except:
        pass


    # _validation that was formerly in the main MCT module, now simply
    # just run it here every time

    if not isinstance(_delete_instr, dict):
        raise TypeError(f"_delete_instr must be a dictionary")

    if len(_delete_instr) != _n_features_in:
        raise ValueError(
            f"_delete_instr must have an entry for each column in X"
        )

    for col_idx, _instr in _delete_instr.items():

        if 'INACTIVE' in _instr and len(_instr) > 1:
            raise ValueError(f"'INACTIVE' IN len(_delete_instr[{col_idx}]) > 1")

        # 'DELETE COLUMN' MUST ALWAYS BE IN THE LAST POSITION!
        if 'DELETE COLUMN' in _instr and _instr[-1] != 'DELETE COLUMN':
            raise ValueError(f"'DELETE COLUMN' IS NOT IN THE -1 POSITION "
                             f"OF _delete_instr[{col_idx}]")

        if len([_ for _ in _instr if _ == 'DELETE COLUMN']) > 1:
            raise ValueError(f"'DELETE COLUMN' IS IN _delete_instr[{col_idx}] "
                             f"MORE THAN ONCE")


    return _delete_instr

































