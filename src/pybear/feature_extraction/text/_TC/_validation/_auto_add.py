# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union



def _val_auto_add(_auto_add: Union[bool, None]) -> None:

    """
    Validate auto_add. Must be boolean or None.


    Parameters
    ----------
    _auto_add:
        Union[bool, None] - Whether to automatically add unknown words
        to the pybear lexicon. pizza revisit this!


    Return
    ------
    -
        None

    """


    if not isinstance(_auto_add, (bool, type(None))):
        raise TypeError(f"'auto_add' must be bool or None")











