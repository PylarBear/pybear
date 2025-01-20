# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from ._core_bool_val import _core_bool_val



def _val_ignore_nan(_ignore_nan: bool) -> bool:

    """
    _ignore_nan must be bool

    """

    return _core_bool_val('_ignore_nan', _ignore_nan)








