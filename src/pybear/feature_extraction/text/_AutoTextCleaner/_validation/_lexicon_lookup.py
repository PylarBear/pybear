# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import LexiconLookupType



def _val_lexicon_lookup(
    _lexicon_lookup: LexiconLookupType
) -> None:


    """
    Validate 'lexicon_lookup'. Must be None or literal 'auto_add',
    'auto_delete', 'manual'.


    Parameters
    ----------
    _lexicon_lookup:
        LexiconLookupType - If None, do not lookup the words in the text
        against the Lexicon. Otherwise, self-explanatory literals that
        indicate how TextLookupRealTime should handle the lookup. For
        a human-less lookup experience, use 'auto_add' or 'auto_delete'.


    Returns
    -------
    -
        None


    """


    if _lexicon_lookup is None:
        return



    err_msg = (f"'lexicon_lookup' must be None, literal string "
               f"'auto_add', auto_delete', or 'manual'.")


    if not isinstance(_lexicon_lookup, str):
        raise TypeError(err_msg)

    if _lexicon_lookup not in ['auto_add', 'auto_delete', 'manual']:
        raise ValueError(err_msg)







