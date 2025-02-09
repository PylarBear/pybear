# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import ShortestWordsType
from typing import Optional

import numbers


def _print_shortest_words(
    shortest_words: ShortestWordsType,
    lp: numbers.Integral,
    rp: numbers.Integral,
    n: Optional[numbers.Integral] = 10
) -> None:

    """
    Print the 'shortest_words_' attribute to screen.


    Parameters
    ----------
    shortest_words:
        dict[str, numbers.Integral] - the dictionary holding the shortest
        strings seen by the fitted TextStatistics instance, and the
        number of characters in the respective string.
    lp:
        numbers.Integral - the left padding for the display.
    rp:
        numbers.Integral - the right padding for the display.
    n:
        Optional[numbers.Integral], default = 10 - the number of shortest
        strings to display.


    Return
    ------
    -
        None

    """


    pass







