# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._TextStatistics._partial_fit. \
    _build_string_frequency import _build_string_frequency

from pybear.feature_extraction.text._TextStatistics._partial_fit. \
    _merge_string_frequency import _merge_string_frequency

from pybear.feature_extraction.text._TextStatistics._partial_fit. \
    _build_overall_statistics_OLD import _build_overall_statistics

from pybear.feature_extraction.text._TextStatistics._partial_fit. \
    _merge_overall_statistics_NOT_USED import _merge_overall_statistics

from pybear.feature_extraction.text._TextStatistics._partial_fit. \
    _build_overall_statistics import _build_overall_statistics_2

from _read_word_bank import _read_word_bank

from pybear.utilities import time_memory_benchmark as tmb



STRINGS = _read_word_bank()



# simulate 2 partial fits with old and new build_overall_statistics.



# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
def version_1(STRINGS):

    # this is the 'old' way of getting overall statistics

    # need to build current overall statistics from STRINGS then merge

    # first partial fit -- -- -- -- -- -- -- -- -- --
    _CURRENT_OVERALL_STATISTICS_DICT = \
        _build_overall_statistics(
            STRINGS,
            case_sensitive=True
        )

    OVERALL_STATISTICS_DICT_1 = \
        _merge_overall_statistics(
            _CURRENT_OVERALL_STATISTICS_DICT,
            {},
            _len_uniques=_CURRENT_OVERALL_STATISTICS_DICT['uniques_count']
            # only because uniques stay the same on both passes
        )
    # END first partial fit -- -- -- -- -- -- -- -- -- --

    # second partial fit -- -- -- -- -- -- -- -- -- -- --
    _CURRENT_OVERALL_STATISTICS_DICT = \
        _build_overall_statistics(
            STRINGS,
            case_sensitive=True
        )

    OVERALL_STATISTICS_DICT_2 = \
        _merge_overall_statistics(
            _CURRENT_OVERALL_STATISTICS_DICT,
            OVERALL_STATISTICS_DICT_1,
            _len_uniques=_CURRENT_OVERALL_STATISTICS_DICT['uniques_count']
            # only because uniques stay the same on both passes
        )
    # END second partial fit -- -- -- -- -- -- -- -- -- --

    return OVERALL_STATISTICS_DICT_2
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

# set these up outside of the function so that it doesnt influence the time.
# in TextStatistics these are already made and are ready to go for the new version.
STRING_FREQUENCY_DICT_1 = \
    _build_string_frequency(STRINGS, case_sensitive=False)

STRING_FREQUENCY_DICT_2 = \
    _merge_string_frequency(STRING_FREQUENCY_DICT_1, STRING_FREQUENCY_DICT_1)


def version_2(STRING_FREQUENCY_DICT_1, STRING_FREQUENCY_DICT_2):

    # this is the 'new' way of getting overall statistics

    # first partial fit -- -- -- -- -- -- -- -- -- --
    OVERALL_STATISTICS_DICT_1 = \
        _build_overall_statistics_2(
            STRING_FREQUENCY_DICT_1,
            case_sensitive=True
        )
    # END first partial fit -- -- -- -- -- -- -- -- -- --

    # second partial fit -- -- -- -- -- -- -- -- -- --
    OVERALL_STATISTICS_DICT_2 = \
        _build_overall_statistics_2(
            STRING_FREQUENCY_DICT_2,
            case_sensitive=True
        )

    return OVERALL_STATISTICS_DICT_2
    # END second partial fit -- -- -- -- -- -- -- -- -- --

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


assert version_1(STRINGS) == version_2(STRING_FREQUENCY_DICT_1, STRING_FREQUENCY_DICT_2)


out = tmb(
    ('version_1', version_1, [STRINGS], {}),
    ('version_2', version_2, [STRING_FREQUENCY_DICT_1, STRING_FREQUENCY_DICT_2], {}),
    number_of_trials=20,
    rest_time=2,
    verbose=True
)





