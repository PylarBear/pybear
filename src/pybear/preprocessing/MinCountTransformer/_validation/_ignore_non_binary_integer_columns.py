# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from ._core_bool_val import _core_bool_val



def _val_ignore_non_binary_integer_columns(
    _inbic: bool
) -> None:

    """
    _ignore_non_binary_integer_columns must be bool


    Parameters
    ----------
    _inbic:
        bool - whether to ignore non-binary integer columns during the
        transform operation.


    Return
    ------
    -
        None

    """


    _core_bool_val('_ignore_non_binary_integer_columns', _inbic)








