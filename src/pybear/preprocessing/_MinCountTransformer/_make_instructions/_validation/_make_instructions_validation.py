# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from ..._type_aliases import (
    CountThresholdType,
    InternalHandleAsBoolType,
    InternalIgnoreColumnsType,
    OriginalDtypesType,
    TotalCountsByColumnType,
    FeatureNamesInType
)
from typing_extensions import Union

from ..._validation._count_threshold import _val_count_threshold
from ..._validation._ignore_columns_handle_as_bool import \
    _val_ignore_columns_handle_as_bool
from ..._validation._original_dtypes import _val_original_dtypes
from ._total_counts_by_column import _val_total_counts_by_column
from ..._validation._feature_names_in import _val_feature_names_in

from ....__shared._validation._any_bool import _val_any_bool
from ....__shared._validation._any_integer import _val_any_integer



def _make_instructions_validation(
    _count_threshold: CountThresholdType,
    _ignore_float_columns: bool,
    _ignore_non_binary_integer_columns: bool,
    _ignore_columns: InternalIgnoreColumnsType,
    _ignore_nan: bool,
    _handle_as_bool: InternalHandleAsBoolType,
    _delete_axis_0: bool,
    _original_dtypes: OriginalDtypesType,
    _n_features_in: int,
    _feature_names_in: Union[FeatureNamesInType, None],
    _total_counts_by_column: TotalCountsByColumnType
) -> None:

    """
    Validate all parameters taken in by _make_instructions. This is a
    centralized hub for validation, see the individual modules for more
    details.


    Parameters
    ----------
    _count_threshold:
        Union[int, Sequence[int]]
    _ignore_float_columns:
        bool
    _ignore_non_binary_integer_columns:
        bool
    _ignore_columns:
        npt.NDArray[np.int32]
    _ignore_nan:
        bool
    _handle_as_bool:
        npt.NDArray[np.int32]
    _delete_axis_0:
        bool
    _original_dtypes:
        OriginalDtypesType
    _n_features_in:
        int
    _feature_names_in:
        Union[FeatureNamesInType, None]
    _total_counts_by_column:
        TotalCountsByColumnType


    Return
    ------
    -
        None


    """


    _val_any_integer(_n_features_in, 'n_features_in', _min=1)

    _val_feature_names_in(
        _feature_names_in,
        _n_features_in
    )

    _val_count_threshold(
        _count_threshold,
        ['int', 'Sequence[int]'],
        _n_features_in
    )

    _val_any_bool(_ignore_float_columns, 'ignore_float_columns', _can_be_None=False)

    _val_any_bool(
        _ignore_non_binary_integer_columns, 'ignore_non_binary_integer_columns',
        _can_be_None=False
    )

    _val_ignore_columns_handle_as_bool(
        _ignore_columns,
        'ignore_columns',
        ['Sequence[int]'],
        _n_features_in=_n_features_in,
        _feature_names_in=_feature_names_in
    )

    _val_any_bool(_ignore_nan, 'ignore_nan', _can_be_None=False)

    _val_original_dtypes(
        _original_dtypes,
        _n_features_in
    )

    _val_ignore_columns_handle_as_bool(
        _handle_as_bool,
        'handle_as_bool',
        ['Sequence[int]'],
        _n_features_in=_n_features_in,
        _feature_names_in=_feature_names_in
    )

    _val_any_bool(_delete_axis_0, 'delete_axis_0', _can_be_None=False)

    _val_total_counts_by_column(_total_counts_by_column)




