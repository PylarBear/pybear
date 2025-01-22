# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from ...._validation._count_threshold import _val_count_threshold
from ...._validation._ignore_float_columns import _val_ignore_float_columns
from ...._validation._ignore_non_binary_integer_columns import \
    _val_ignore_non_binary_integer_columns
from ...._validation._core_ign_cols_handle_as_bool import _core_ign_cols_handle_as_bool
from ...._validation._ignore_nan import _val_ignore_nan
from ...._validation._delete_axis_0 import _val_delete_axis_0
from ...._validation._handle_as_bool_v_dtypes import _val_handle_as_bool_v_dtypes
from ._original_dtypes import _val_original_dtypes
from ._total_counts_by_column import _val_total_counts_by_column
from ._n_features_in import _val_n_features_in

from typing_extensions import Union
import numpy.typing as npt

import numpy as np

from ...._type_aliases import (
    OriginalDtypesDtype,
    TotalCountsByColumnType
)



def _make_instructions_validation(
    _count_threshold: int,
    _ignore_float_columns: bool,
    _ignore_non_binary_integer_columns: bool,
    _ignore_columns: npt.NDArray[int],
    _ignore_nan: bool,
    _handle_as_bool: npt.NDArray[int],
    _delete_axis_0: bool,
    _original_dtypes: OriginalDtypesDtype,
    _n_features_in: int,
    _feature_names_in: Union[npt.NDArray[object], None],
    _total_counts_by_column: TotalCountsByColumnType,
    _threshold: Union[int, None] = None
) -> tuple[npt.NDArray[int], npt.NDArray[int]]:

    """
    Validate all args/kwargs taken in by _make_instructions. This
    consolidates code within _make_instructions.

    Parameters
    ----------
    _count_threshold: int
    _ignore_float_columns: bool
    _ignore_non_binary_integer_columns: bool,
    _ignore_columns: np.ndarray[int],
    _ignore_nan: bool,
    _handle_as_bool: np.ndarray[int],
    _delete_axis_0: bool,
    _original_dtypes: OriginalDtypesDtype,
    _n_features_in: int,
    _feature_names_in: Union[npt.NDArray[object], None],
    _total_counts_by_column: TotalCountsByColumnType,
    _threshold: Union[int, None] = None

    Return
    ------
    -
        _ignore_columns: np.ndarray[int]
        _handle_as_bool: np.ndarray[int]


    """


    _val_count_threshold(_count_threshold)

    _val_ignore_float_columns(_ignore_float_columns)

    _val_ignore_non_binary_integer_columns(_ignore_non_binary_integer_columns)

    _val_n_features_in(_n_features_in)

    _ignore_columns = _core_ign_cols_handle_as_bool(
        _ignore_columns,
        'ignore_columns',
        _mct_has_been_fit=True,
        _n_features_in=_n_features_in,
        _feature_names_in=_feature_names_in
    )

    _val_ignore_nan(_ignore_nan)

    _val_original_dtypes(_original_dtypes)

    _handle_as_bool = _core_ign_cols_handle_as_bool(
        _handle_as_bool,
        'handle_as_bool',
        _mct_has_been_fit=True,
        _n_features_in=_n_features_in,
        _feature_names_in=_feature_names_in
    )

    _val_handle_as_bool_v_dtypes(_handle_as_bool, _original_dtypes)

    _val_delete_axis_0(_delete_axis_0)

    _val_total_counts_by_column(_total_counts_by_column)

    _val_count_threshold(_threshold or _count_threshold)


    return _ignore_columns, _handle_as_bool














