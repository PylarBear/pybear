# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _val_update_lexicon(_update_lexicon: bool) -> None:

    """
    Validate update_lexicon. Must be boolean.


    Parameters
    ----------
    _update_lexicon:
        bool - Whether to stage unknown words for addition to the pybear
        Lexicon. If 'update_lexicon' is True and 'auto_add_to_lexicon' is True,
        the words will be silently staged in the LEXICON_ADDENDUM. If
        'update_lexicon' is True and 'auto_add_to_lexicon' is False, the user will
        be prompted whether to add the words to the LEXICON_ADDENDUM.



    Return
    ------
    -
        None

    """


    if not isinstance(_update_lexicon, bool):
        raise TypeError(f"'update_lexicon' must be boolean")











