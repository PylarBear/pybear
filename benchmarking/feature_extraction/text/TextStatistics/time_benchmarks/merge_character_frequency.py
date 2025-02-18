# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from _read_word_bank import _read_word_bank

from pybear.feature_extraction.text._TextStatistics._partial_fit. \
    _build_string_frequency import _build_string_frequency

from pybear.feature_extraction.text._TextStatistics._partial_fit. \
    _build_character_frequency import _build_character_frequency

from pybear.utilities import time_memory_benchmark as tmb


# LINUX
# type_1     time = 0.000 +/- 0.000 sec; mem = 0.000 +/- 0.000 MB
# type_2     time = 0.000 +/- 0.000 sec; mem = 0.000 +/- 0.000 MB



# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
_raw_strings = _read_word_bank()
_single_strings = []
for line in _raw_strings:
    _single_strings += (line.split(sep=' '))
del _raw_strings

single_string_frequency = _build_string_frequency(
    _single_strings,
    case_sensitive=False
)

character_frequency = _build_character_frequency(single_string_frequency)
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --



# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
def type_1(_string_frequency):

    _character_frequency = {}

    for _string, _ct in _string_frequency.items():
        for _char in str(_string):
            _character_frequency[_char] = \
                (_character_frequency.get(_char, 0) + _ct)

    _character_frequency = dict((zip(
        map(str, _character_frequency.keys()),
        map(int, _character_frequency.values())
    )))


    return _character_frequency


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
def type_1(
    _current_character_frequency,
    _character_frequency
):

    for k, v in _current_character_frequency.items():

        _character_frequency[str(k)] = (_character_frequency.get(str(k), 0) + v)

    return _character_frequency
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
def type_2(
    _current_character_frequency,
    _character_frequency
):

    for k, v in _current_character_frequency.items():

        if k in _character_frequency:
            _character_frequency[k] += v
        else:
            _character_frequency[k] = v


    return _character_frequency
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


assert type_1(character_frequency, character_frequency) == \
       type_2(character_frequency, character_frequency)


out = tmb(
    ('type_1', type_1, [character_frequency, character_frequency], {}),
    ('type_2', type_2, [character_frequency, character_frequency], {}),
    number_of_trials=10,
    rest_time=1,
    verbose=True
)



