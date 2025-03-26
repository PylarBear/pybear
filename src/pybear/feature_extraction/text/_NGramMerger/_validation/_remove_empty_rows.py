# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _val_remove_empty_rows(_remove_empty_rows: bool) -> None:

    """
    Validate remove_empty_rows. Must be boolean.


    Parameters
    ----------
    _remove_empty_rows:
        bool - whether to delete any empty rows that may occur during
        the merging process. A row could only become empty if 'wrap' is
        True.


    Returns
    -------
    -
        None


    """


    if not isinstance(_remove_empty_rows, bool):
        raise TypeError(f"'remove_empty_rows' must be boolean")





