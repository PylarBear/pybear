# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from ._core_bool_val import _core_bool_val



def _val_delete_axis_0(_delete_axis_0: bool) -> None:

    """
    _delete_axis_0 must be bool


    Parameters
    ----------
    _delete_axis_0:
        bool - whether to delete the rows associated with values that
        are below threshold for binary or 'handle_as_bool' columns.
        Normally, a binary or 'handle_as_bool' column would be deleted
        in entirety without deleting the corresponding rows if one of
        the two values is below threshold.


    Return
    ------
    -
        None


    """


    _core_bool_val('_delete_axis_0', _delete_axis_0)








