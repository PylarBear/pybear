# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional

import numbers

from .._validation._string_frequency import _val_string_frequency
from .._validation._n import _val_n

from .._get._get_shortest_strings import _get_shortest_strings



def _print_shortest_strings(
    string_frequency: dict[str, numbers.Integral],
    lp: numbers.Integral,
    rp: numbers.Integral,
    n: Optional[numbers.Integral] = 10
) -> None:
    """Print the shortest strings in the `string_frequency_` attribute
    and their frequencies to screen.

    Only available if TS parameter `store_uniques` is True. If False,
    `string_frequency` is empty, so print a message that uniques are not
    available.

    Parameters
    ----------
    string_frequency : dict[str, numbers.Integral]
        The dictionary holding all the unique strings and their
        frequencies seen by the fitted `TextStatistics` instance.
    lp : numbers.Integral
        The left padding for the display.
    rp : numbers.Integral
        The right padding for the display.
    n : Optional[numbers.Integral], default = 10
        The number of shortest strings to print to screen.

    Returns
    -------
    None

    """


    _val_string_frequency(string_frequency)


    if not len(string_frequency):
        print(
            "Parameter 'store_uniques' is False, individual uniques have "
            "not been retained for display."
        )
        return

    _val_n(n)


    n = min(n, len(string_frequency))

    shortest_string_dict = _get_shortest_strings(string_frequency, n)


    print(f'\n TOP {n} SHORTEST STRINGS OF {len(string_frequency)}:')

    _max_len = max(map(len, shortest_string_dict.keys()))

    print(lp * ' ' + (f'STRING').ljust(_max_len + rp) + f'FREQUENCY')
    for k, v in shortest_string_dict.items():
        print(lp * ' ' + f'{k}'.ljust(_max_len + rp) +f'{v}')

    del shortest_string_dict, _max_len







