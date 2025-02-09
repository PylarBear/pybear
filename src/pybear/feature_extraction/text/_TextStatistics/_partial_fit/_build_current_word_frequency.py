# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence, Optional
from .._type_aliases import WordFrequencyType

import numpy as np

from .._validation._words import _val_words



def _build_current_word_frequency(
    WORDS: Sequence[str],
    case_sensitive: Optional[bool] = False
) -> WordFrequencyType:

    """
    Build a dictionary of the unique words in WORDS and their counts.


    Parameters
    ----------
    WORDS:
        Sequence[str] - the sequence of words currently being fitted.
    case_sensitive:
        Optional[bool], default = False - whether to preserve the case
        of the characters when getting the uniques. When False, normalize
        the case of all characters.


    Return
    ------
    -
        dict[str, int] - a dictionary with the unique words in WORDS as
        keys and their respective counts as values.


    """


    _val_words(WORDS)

    assert isinstance(case_sensitive, bool)

    if case_sensitive:
        _word_frequency = dict((zip(*np.unique(WORDS, return_counts=True))))
    elif not case_sensitive:
        _word_frequency = dict((zip(
            *np.unique(list(map(str.upper, WORDS)), return_counts=True)
        )))

    _word_frequency = dict((zip(
        map(str, _word_frequency.keys()),
        map(int, _word_frequency.values())
    )))


    return _word_frequency






