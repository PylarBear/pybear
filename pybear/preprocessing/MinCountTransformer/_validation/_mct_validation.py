# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

# PIZZA FINISH

from typing import Union

from .._type_aliases import (
    DataType,
    OriginalDtypesDtype,
    TotalCountsByColumnType,
    InstructionsType
)

from .._shared._validation._val_ignore_float_columns import \
    _val_ignore_float_columns
from .._shared._validation._val_ignore_non_binary_integer_columns import \
    _val_ignore_non_binary_integer_columns
from .._shared._validation._val_ignore_nan import _val_ignore_nan
from .._shared._validation._val_delete_axis_0 import _val_delete_axis_0
from .._shared._validation._val_reject_unseen_values import \
    _val_reject_unseen_values
from .._shared._validation._val_max_recursions import _val_max_recursions
from .._shared._validation._val_n_jobs import _val_n_jobs




def _mct_validation(
        _count_threshold: int,
        _ignore_float_columns: bool,
        _ignore_non_binary_integer_columns: bool,
        _ignore_nan: bool,
        _delete_axis_0: bool,
        _ignore_columns: [list[Union[int, str]], callable, None],
        _handle_as_bool: [list[Union[str, int]], callable, None],
        _reject_unseen_values: bool,
        _max_recursions: int,
        _n_jobs: Union[int, None]
    ) -> tuple[int, bool, bool, [list[Union[int, str]], callable, None], \
            [list[Union[int, str]], callable, None], bool ,int, int]:


    """
    Validate arg and kwargs for MinCountTransformer.
    
    Parameters
    ----------
    
    
    
    Return
    ------
    
    
    
    """


    # pizza move imports when finished
    from ._shared._validation._val_count_threshold import _val_count_threshold

    # core count_threshold val
    _count_threshold = _val_count_threshold(_count_threshold)

    _ignore_float_columns = _val_ignore_float_columns(_ignore_float_columns)

    _ignore_non_binary_integer_columns = _val_ignore_non_binary_integer_columns(
        _ignore_non_binary_integer_columns
    )

    _ignore_nan = _val_ignore_nan(_ignore_nan)

    _delete_axis_0 = _val_delete_axis_0(_delete_axis_0)

    # pizza figure this out, it's complicated
    _ignore_columns = \
        _val_ign_cols_handle_as_bool(_ignore_columns, 'ignore_columns')
    _handle_as_bool = \
        _val_ign_cols_handle_as_bool(_handle_as_bool, 'handle_as_bool')

    _reject_unseen_values = _val_reject_unseen_values(_reject_unseen_values)

    _max_recursions = _val_max_recursions(_max_recursions)

    _n_jobs = _val_n_jobs(_n_jobs)




    return _count_threshold, _ignore_float_columns, _ignore_non_binary_integer_columns, \
        _ignore_nan, _delete_axis_0, _ignore_columns, _handle_as_bool, \
        _reject_unseen_values, _max_recursions, _n_jobs









