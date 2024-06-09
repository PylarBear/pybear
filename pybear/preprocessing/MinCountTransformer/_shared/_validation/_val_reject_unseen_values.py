# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from ._core_bool_val import _core_bool_val



def _val_reject_unseen_values(_reject_unseen_values: bool) -> bool:

    """
    _reject_unseen_values must be bool

    """

    return _core_bool_val('_reject_unseen_values', _reject_unseen_values)








