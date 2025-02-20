# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union



def _val_update_lexicon(_update_lexicon: Union[bool, None]) -> None:

    """
    Validate update_lexicon. Must be boolean or None.


    Parameters
    ----------
    _update_lexicon:
        Union[bool, None] - Whether to automatically add unknown words
        to the pybear lexicon. pizza revisit this!


    Return
    ------
    -
        None

    """


    if not isinstance(_update_lexicon, (bool, type(None))):
        raise TypeError(f"'update_lexicon' must be bool or None")











