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
        bool - Whether to automatically delete any empty rows that may
        be left after the stop word removal process.


    Return
    ------
    -
        None

    """


    if not isinstance(_remove_empty_rows, bool):
        raise TypeError(f"'remove_empty_rows' must be boolean")











