# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._TextStatistics._print._string_frequency \
    import _print_string_frequency

from pybear.feature_extraction.text._TextStatistics._partial_fit. \
    _build_string_frequency import _build_string_frequency

from _read_green_eggs_and_ham import _read_green_eggs_and_ham



STRINGS = _read_green_eggs_and_ham()

string_frequency = \
    _build_string_frequency(
        STRINGS,
        case_sensitive=True
    )

_print_string_frequency(
    string_frequency,
    lp=5,
    rp=15,
    n=100
)




