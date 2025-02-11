# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._TextStatistics._print._character_frequency \
    import _print_character_frequency

from pybear.feature_extraction.text._TextStatistics._partial_fit. \
    _build_string_frequency import _build_string_frequency

from pybear.feature_extraction.text._TextStatistics._partial_fit. \
    _build_character_frequency import _build_character_frequency

from _read_green_eggs_and_ham import _read_green_eggs_and_ham



STRINGS = _read_green_eggs_and_ham()


character_frequency = _build_character_frequency(
    _build_string_frequency(STRINGS, case_sensitive=True)
)

_print_character_frequency(
    character_frequency,
    lp=5,
    rp=15
)





