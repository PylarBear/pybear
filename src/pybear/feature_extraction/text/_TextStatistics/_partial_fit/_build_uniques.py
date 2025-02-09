# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import (
    Sequence,
    Optional
)

from .._validation._words import _val_words


# pizza this may not be needed, _word_frequency is getting uniques also



def _build_uniques(
    WORDS: Sequence[str],
    case_sensitive: Optional[bool] = False
) -> Sequence[str]:

    """
    Build a sequence of unique strings from the sequance of strings in
    'WORDS'.


    Parameters
    ----------
    WORDS:
        Sequence[str] - a sequence of strings.
    case_sensitive:
        Optional[bool], default = False - whether to observe the
        capitalization of the strings in 'WORDS' or normalize all the
        strings to the same case.


    Return
    ------
    -
        _uniques: Sequence[str] - the unique strings in 'WORDS'.

    """


    _val_words(WORDS)

    if not isinstance(case_sensitive, bool):
        raise TypeError(f"'case_sensitive' must be boolean")

    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # pizza make a decision on the format of UNIQUES
    if case_sensitive:
        _uniques = set(WORDS)
    elif not case_sensitive:
        _uniques = set(map(str.upper, WORDS))


    return _uniques

