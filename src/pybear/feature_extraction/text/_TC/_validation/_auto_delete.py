# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union



def _val_auto_delete(_auto_delete: Union[bool, None]) -> None:

    """
    Validate auto_delete. Must be boolean or None.


    Parameters
    ----------
    _auto_delete:
        Union[bool, None] - Whether to automatically delete unknown words
        from the data (not the lexicon). pizza revisit this.


    Return
    ------
    -
        None

    """


    if not isinstance(_auto_delete, (bool, type(None))):
        raise TypeError(f"'auto_delete' must be bool or None")











