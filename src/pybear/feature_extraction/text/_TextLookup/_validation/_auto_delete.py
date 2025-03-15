# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _val_auto_delete(_auto_delete: bool) -> None:

    """
    Validate auto_delete. Must be boolean.


    Parameters
    ----------
    _auto_delete:
        bool - Whether to automatically delete unknown words from the
        data (not the Lexicon).


    Return
    ------
    -
        None

    """


    if not isinstance(_auto_delete, bool):
        raise TypeError(f"'auto_delete' must be boolean")











