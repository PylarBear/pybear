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
    TotalCountsByColumnType
)
from typing_extensions import Union
import numpy.typing as npt

from ..._validation._count_threshold import _val_count_threshold
from ..._validation._ignore_float_columns import _val_ignore_float_columns
from ..._validation._ignore_non_binary_integer_columns import \
    _val_ignore_non_binary_integer_columns
from ..._validation._ignore_columns_handle_as_bool import \
    _val_ignore_columns_handle_as_bool
from ..._validation._ignore_nan import _val_ignore_nan
from ..._validation._delete_axis_0 import _val_delete_axis_0
from ..._validation._original_dtypes import _val_original_dtypes
from ._total_counts_by_column import _val_total_counts_by_column
from ..._validation._n_features_in import _val_n_features_in
from ..._validation._feature_names_in import _val_feature_names_in



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
    _feature_names_in: Union[npt.NDArray[object], None],
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
        Union[npt.NDArray[object], None]
    _total_counts_by_column:
        TotalCountsByColumnType


    Return
    ------
    -
        None


    """


    _val_n_features_in(_n_features_in)

    _val_feature_names_in(
        _feature_names_in,
        _n_features_in
    )

    _val_count_threshold(
        _count_threshold,
        ['int', 'Sequence[int]'],
        _n_features_in
    )

    _val_ignore_float_columns(_ignore_float_columns)

    _val_ignore_non_binary_integer_columns(_ignore_non_binary_integer_columns)

    _val_ignore_columns_handle_as_bool(
        _ignore_columns,
        'ignore_columns',
        ['Sequence[int]'],
        _n_features_in=_n_features_in,
        _feature_names_in=_feature_names_in
    )

    _val_ignore_nan(_ignore_nan)

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

    _val_delete_axis_0(_delete_axis_0)

    _val_total_counts_by_column(_total_counts_by_column)















