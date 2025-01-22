# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from ._core_bool_val import _core_bool_val



def _val_reject_unseen_values(
    _reject_unseen_values: bool
) -> bool:

    """
    _reject_unseen_values must be bool


    Parameters
    ----------
    _reject_unseen_values:
        bool - when transforming data, whether to raise if the column
            search finds unique values that were not seen during fitting.


    Return
    ------
    -
        None

    """

    return _core_bool_val('_reject_unseen_values', _reject_unseen_values)








