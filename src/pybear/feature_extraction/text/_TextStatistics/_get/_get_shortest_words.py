# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional
from .._type_aliases import WordFrequencyType

import numbers

import numpy as np

from .._validation._word_frequency import _val_word_frequency
from .._validation._n import _val_n



def _get_shortest_words(
    word_frequency: WordFrequencyType,
    n: Optional[numbers.Integral] = 10
) -> dict[str, numbers.Integral]:

    """
    Return the shortest strings in the 'word_frequency_' attribute as a
    dictionary with strings as keys and frequencies as values.


    Parameters
    ----------
    word_frequency:
        dict[str, numbers.Integral] - the dictionary holding the unique
        strings seen by the fitted TextStatistics instance, and the
        number of occurrences of each string.
    n:
        Optional[numbers.Integral], default = 10 - the number of top
        shortest strings to retrieve.


    Return
    ------
    -
        shortest_strings: dict[str, numbers.Integral] - the top 'n'
        shortest strings and their frequencies.

    """


    _val_word_frequency(word_frequency)
    _val_n(n)


    _LENS = np.fromiter(map(len, word_frequency), dtype=np.uint32)
    _UNIQUES = np.fromiter(word_frequency.keys(), dtype=f"<U{int(np.max(_LENS))}")
    _COUNTS = np.fromiter(word_frequency.values(), dtype=np.uint32)

    n = min(n, len(_UNIQUES))
    # SORT ON len(str) FIRST, THEN ASC ALPHA ON STR (lexsort GOES BACKWARDS)
    MASK = np.lexsort((_UNIQUES, _LENS))[:n]
    del _LENS

    TOP_SHORTEST_STRINGS = _UNIQUES[MASK]
    TOP_FREQUENCIES = _COUNTS[MASK]
    del _UNIQUES, _COUNTS, MASK

    shortest_strings = dict((zip(
        map(str, TOP_SHORTEST_STRINGS),
        map(int, TOP_FREQUENCIES)
    )))

    del TOP_SHORTEST_STRINGS, TOP_FREQUENCIES


    return shortest_strings









