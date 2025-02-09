# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import WordFrequencyType
from typing import Optional

import numbers



def _print_longest_words(
    longest_words: WordFrequencyType,
    lp: numbers.Integral,
    rp: numbers.Integral,
    n: Optional[numbers.Integral] = 10
) -> None:

    """
    Print the 'longest_words_' attribute to screen.


    Parameters
    ----------
    longest_words:
        dict[str, numbers.Integral] - the dictionary holding the longest
        strings seen by the fitted TextStatistics instance, and the
        number of characters in the respective string.
    lp:
        numbers.Integral - the left padding for the display.
    rp:
        numbers.Integral - the right padding for the display.
    n:
        Optional[numbers.Integral], default = 10 - the number of top
        longest strings to display.


    Return
    ------
    -
        None

    """

    n = min(20, len(_UNIQUES))
    print(f'\nTOP {n} LONGEST WORDS:')

    _LENS = np.fromiter(map(len, _UNIQUES), dtype=np.int8)

    MASK = np.flip(np.argsort(_LENS))
    LONGEST_WORDS = _UNIQUES[MASK][:n]
    _LENS = _LENS[MASK][:n]
    del MASK

    print(_lp * ' ' + f'WORD'.ljust(3 * _rp) + f'LENGTH')
    for i in range(n):
        print(_lp * ' ' + f'{(LONGEST_WORDS[i])}'.ljust(3 * _rp), end='')
        print(f'{_LENS[i]}')

    del LONGEST_WORDS










