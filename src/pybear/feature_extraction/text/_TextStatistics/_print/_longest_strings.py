# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import StringFrequencyType
from typing import Optional

import numbers

from .._validation._string_frequency import _val_string_frequency
from .._validation._n import _val_n

from .._get._get_longest_strings import _get_longest_strings



def _print_longest_strings(
    string_frequency: StringFrequencyType,
    lp: numbers.Integral,
    rp: numbers.Integral,
    n: Optional[numbers.Integral] = 10
) -> None:

    """
    Print the longest strings in the 'string_frequency_' attribute and
    their frequencies to screen.


    Parameters
    ----------
    string_frequency:
        dict[str, numbers.Integral] - the dictionary holding all the
        unique strings and their frequencies seen by the fitted
        TextStatistics instance.
    lp:
        numbers.Integral - the left padding for the display.
    rp:
        numbers.Integral - the right padding for the display.
    n:
        Optional[numbers.Integral], default = 10 - the number of longest
        strings to print to screen.


    Return
    ------
    -
        None

    """


    _val_string_frequency(string_frequency)
    _val_n(n)


    n = min(n, len(string_frequency))

    longest_string_dict = _get_longest_strings(string_frequency, n)


    print(f'\n TOP {n} LONGEST STRINGS OF {len(string_frequency)}:')

    _max_len = max(map(len, longest_string_dict.keys()))

    print(lp * ' ' + (f'STRING').ljust(_max_len + rp) + f'FREQUENCY')
    for k, v in longest_string_dict.items():
        print(lp * ' ' + f'{k}'.ljust(_max_len + rp) +f'{v}')

    del longest_string_dict, _max_len







