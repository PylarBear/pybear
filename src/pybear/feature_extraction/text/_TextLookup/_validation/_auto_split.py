# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _val_auto_split(_auto_split: bool) -> None:

    """
    Validate auto_split. Must be boolean.


    Parameters
    ----------
    _auto_split:
        bool - Whether to automatically delete unknown words from the
        data (not the Lexicon).


    Return
    ------
    -
        None

    """


    if not isinstance(_auto_split, bool):
        raise TypeError(f"'auto_split' must be boolean")











