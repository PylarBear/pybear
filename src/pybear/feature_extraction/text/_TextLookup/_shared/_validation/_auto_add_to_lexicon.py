# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _val_auto_add_to_lexicon(_auto_add_to_lexicon: bool) -> None:

    """
    Validate auto_add_to_lexicon. Must be boolean.


    Parameters
    ----------
    _auto_add_to_lexicon:
        bool - Whether to automatically stage unknown words for addition
        to the pybear Lexicon.


    Return
    ------
    -
        None

    """


    if not isinstance(_auto_add_to_lexicon, bool):
        raise TypeError(f"'auto_add_to_lexicon' must be boolean")











