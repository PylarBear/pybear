# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import WordFrequencyType
from typing import Optional

import numbers

from .._validation._word_frequency import _val_word_frequency
from .._validation._n import _val_n

from .._get._get_shortest_words import _get_shortest_words



def _print_shortest_words(
    word_frequency: WordFrequencyType,
    lp: numbers.Integral,
    rp: numbers.Integral,
    n: Optional[numbers.Integral] = 10
) -> None:

    """
    Print the shortest strings in the 'word_frequency_' attribute and
    their frequencies to screen.


    Parameters
    ----------
    word_frequency:
        dict[str, numbers.Integral] - the dictionary holding all the
        unique strings and their frequencies seen by the fitted
        TextStatistics instance.
    lp:
        numbers.Integral - the left padding for the display.
    rp:
        numbers.Integral - the right padding for the display.
    n:
        Optional[numbers.Integral], default = 10 - the number of shortest
        strings to print to screen.


    Return
    ------
    -
        None

    """


    _val_word_frequency(word_frequency)
    _val_n(n)


    n = min(n, len(word_frequency))

    shortest_string_dict = _get_shortest_words(word_frequency, n)


    print(f'\n TOP {n} SHORTEST STRINGS OF {len(word_frequency)}:')

    _max_len = max(map(len, shortest_string_dict.keys()))

    print(lp * ' ' + (f'STRING').ljust(_max_len + rp) + f'FREQUENCY')
    for k, v in shortest_string_dict.items():
        print(lp * ' ' + f'{k}'.ljust(_max_len + rp) +f'{v}')

    del shortest_string_dict, _max_len







