# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._TextStatistics._print._longest_strings \
    import _print_longest_strings

from pybear.feature_extraction.text._TextStatistics._partial_fit. \
    _build_string_frequency import _build_string_frequency

from _read_green_eggs_and_ham import _read_green_eggs_and_ham


STRINGS = _read_green_eggs_and_ham()

string_frequency = \
    _build_string_frequency(
        STRINGS,
        case_sensitive=True
    )

_print_longest_strings(
    string_frequency,
    lp=5,
    rp=15,
    n=10
)

print(f'-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- ')

_print_longest_strings(
    {},
    lp=5,
    rp=15,
    n=10
)
