# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._TextStatistics._print._startswith_frequency  \
    import _print_starts_with_frequency

from pybear.feature_extraction.text._TextStatistics._partial_fit. \
    _build_string_frequency import _build_string_frequency

from pybear.feature_extraction.text._TextStatistics._partial_fit. \
    _build_startswith_frequency import _build_startswith_frequency

from _read_green_eggs_and_ham import _read_green_eggs_and_ham



STRINGS = _read_green_eggs_and_ham()


starts_with_frequency = _build_startswith_frequency(
    _build_string_frequency(STRINGS, case_sensitive=True)
)


_print_starts_with_frequency(
    starts_with_frequency,
    lp=5,
    rp=15
)



