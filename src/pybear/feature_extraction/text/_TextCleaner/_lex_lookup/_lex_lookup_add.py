# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from copy import deepcopy


def _lex_lookup_add(
        _word:str,
        _LEXICON_ADDENDUM:list[str],
        _KNOWN_WORDS:list[str]
    ) -> tuple[list[str], list[str]]:


    """
    Append a word to the LEXICON_ADDENDUM and KNOWN_WORDS objects.

    Parameters
    ----------
    word:
        str - the word to append to LEXICON_ADDENDUM and KNOWN_WORDS
    LEXICON_ADDENDUM:
        list[str] - list of words to be reported for manual insert into
        Lexicon
    KNOWN_WORDS:list[str]
        list[str] - a working list of words active during the current
        TextCleaner instance that stores words that are to be handled as
        if they are in the fixed Lexicon.

    Return
    ------
     LEXICON_ADDENDUM:
        list[str] - appended LEXICON_ADDENDUM
    KNOWN_WORDS:list[str]
        list[str] - appended KNOWN_WORDS

    """

    if not isinstance(_word, str):
        raise TypeError(f"'word' must be a string")

    _word = _word.upper()

    if not isinstance(_LEXICON_ADDENDUM, list):
        raise TypeError(f"'LEXICON_ADDENDUM' must be a list of strings")

    if not all(map(isinstance, _LEXICON_ADDENDUM, (str for _ in _LEXICON_ADDENDUM))):
        raise TypeError(f"'LEXICON_ADDENDUM' must be a list of strings")

    if not isinstance(_KNOWN_WORDS, list):
        raise TypeError(f"'KNOWN_WORDS' must be a list of strings")

    if not all(map(isinstance, _KNOWN_WORDS, (str for _ in _KNOWN_WORDS))):
        raise TypeError(f"'KNOWN_WORDS' must be a list of strings")


    __LEXICON_ADDENDUM = deepcopy(_LEXICON_ADDENDUM)
    __KNOWN_WORDS = deepcopy(_KNOWN_WORDS)


    __LEXICON_ADDENDUM.append(_word)
    __KNOWN_WORDS.append(_word)

    __LEXICON_ADDENDUM.sort()
    __KNOWN_WORDS.sort()

    return __LEXICON_ADDENDUM, __KNOWN_WORDS








