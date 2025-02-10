# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._TextStatistics._print._shortest_words \
    import _print_shortest_words

from pybear.feature_extraction.text._TextStatistics._partial_fit. \
    _build_word_frequency import _build_word_frequency

from _read_green_eggs_and_ham import _read_green_eggs_and_ham


STRINGS = _read_green_eggs_and_ham()

word_frequency = \
    _build_word_frequency(
        STRINGS,
        case_sensitive=True
    )

_print_shortest_words(
    word_frequency,
    lp=5,
    rp=15,
    n=10
)



