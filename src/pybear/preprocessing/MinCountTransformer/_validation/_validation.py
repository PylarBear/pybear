# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Callable
from typing_extensions import Union

from .._type_aliases import IgnColsHandleAsBoolDtype

import numpy as np

from .._validation._val_count_threshold import _val_count_threshold
from .._validation._val_ignore_float_columns import _val_ignore_float_columns
from .._validation._val_ignore_non_binary_integer_columns import \
    _val_ignore_non_binary_integer_columns
from .._validation._val_ignore_nan import _val_ignore_nan
from .._validation._val_delete_axis_0 import _val_delete_axis_0
from .._validation._val_ignore_columns import _val_ignore_columns
from .._validation._val_handle_as_bool import _val_handle_as_bool
from .._validation._val_reject_unseen_values import _val_reject_unseen_values
from .._validation._val_max_recursions import _val_max_recursions
from .._validation._val_n_jobs import _val_n_jobs




def _validation(
    _count_threshold: int,
    _ignore_float_columns: bool,
    _ignore_non_binary_integer_columns: bool,
    _ignore_nan: bool,
    _delete_axis_0: bool,
    _ignore_columns: IgnColsHandleAsBoolDtype,
    _handle_as_bool: IgnColsHandleAsBoolDtype,
    _reject_unseen_values: bool,
    _max_recursions: int,
    _n_jobs: Union[int, None],
    _mct_has_been_fit: bool=False,
    _n_features_in: Union[int, None]=None,
    _feature_names_in: Union[np.ndarray[str], None]=None,
    _original_dtypes: Union[np.ndarray[str], None]=None
) -> tuple[
    int,
    bool,
    bool,
    bool,
    bool,
    Union[np.ndarray[int], np.ndarray[str], Callable[[np.ndarray], np.ndarray]],
    Union[np.ndarray[int], np.ndarray[str], Callable[[np.ndarray], np.ndarray]],
    bool,
    int,
    int
]:


    """
    Validate arg and kwargs for MinCountTransformer.
    
    Parameters
    ----------
    _count_threshold: int,
    _ignore_float_columns: bool,
    _ignore_non_binary_integer_columns: bool,
    _ignore_nan: bool,
    _delete_axis_0: bool,
    _ignore_columns: IgnColsHandleAsBoolDtype,
    _handle_as_bool: IgnColsHandleAsBoolDtype,
    _reject_unseen_values: bool,
    _max_recursions: int,
    _n_jobs: Union[int, None],
    _mct_has_been_fit: bool=False,
    _n_features_in: Union[int, None]=None,
    _feature_names_in: Union[np.ndarray[str], None]=None,
    _original_dtypes: Union[np.ndarray[str], None]=None
    
    
    Return
    ------
    -
        _count_threshold: int
        _ignore_float_columns: bool
        _ignore_non_binary_integer_columns: bool
        _ignore_nan: bool
        _delete_axis_0: bool
        _ignore_columns: np.ndarray
        _handle_as_bool: np.ndarray
        _reject_unseen_values: bool
        _max_recursions: int
        _n_jobs: int
    
    
    """


    # core count_threshold val
    _count_threshold = _val_count_threshold(_count_threshold)

    _ignore_float_columns = _val_ignore_float_columns(_ignore_float_columns)

    _ignore_non_binary_integer_columns = _val_ignore_non_binary_integer_columns(
        _ignore_non_binary_integer_columns
    )

    _ignore_nan = _val_ignore_nan(_ignore_nan)

    _delete_axis_0 = _val_delete_axis_0(_delete_axis_0)

    _ignore_columns = _val_ignore_columns(
        _ignore_columns,
        _mct_has_been_fit=_mct_has_been_fit,
        _n_features_in=_n_features_in,
        _feature_names_in=_feature_names_in
    )

    _handle_as_bool = _val_handle_as_bool(
        _handle_as_bool,
        _mct_has_been_fit=_mct_has_been_fit,
        _n_features_in=_n_features_in,
        _feature_names_in=_feature_names_in,
        _original_dtypes=_original_dtypes
    )

    _reject_unseen_values = _val_reject_unseen_values(_reject_unseen_values)

    _max_recursions = _val_max_recursions(_max_recursions)

    _n_jobs = _val_n_jobs(_n_jobs)




    return _count_threshold, _ignore_float_columns, _ignore_non_binary_integer_columns, \
        _ignore_nan, _delete_axis_0, _ignore_columns, _handle_as_bool, \
        _reject_unseen_values, _max_recursions, _n_jobs









