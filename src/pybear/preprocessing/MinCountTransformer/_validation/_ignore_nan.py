# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from ._core_bool_val import _core_bool_val



def _val_ignore_nan(
    _ignore_nan: bool
) -> bool:

    """
    _ignore_nan must be bool


    Parameters
    ----------
    _ignore_nan:
        bool - whether to ignore nan-like values during the column
        search or include them as a unique value that is subject to the
        minimum threshold rules.


    Return
    ------
    -
        None


    """

    return _core_bool_val('_ignore_nan', _ignore_nan)








