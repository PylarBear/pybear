# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from ._core_bool_val import _core_bool_val



def _val_ignore_float_columns(_ignore_float_columns: bool) -> bool:

    """
    _ignore_float_columns must be bool

    """


    return _core_bool_val('ignore_float_columns', _ignore_float_columns)








