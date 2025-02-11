# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from _read_word_bank import _read_word_bank

from pybear.feature_extraction.text._TextStatistics._partial_fit. \
    _build_string_frequency import _build_string_frequency

from pybear.utilities import time_memory_benchmark as tmb



# LINUX
# type_1a     time = 0.034 +/- 0.001 sec; mem = 0.000 +/- 0.000 MB
# type_1b     time = 0.572 +/- 0.004 sec; mem = 0.000 +/- 0.000 MB
# type_2a     time = 0.040 +/- 0.001 sec; mem = 0.000 +/- 0.000 MB
# type_2b     time = 0.673 +/- 0.009 sec; mem = 0.000 +/- 0.000 MB

# the bottleneck is long strings in string_frequency dict.
# character_frequency has to iterate over the long string and put each
# character in its character bucket.



# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
_long_strings = _read_word_bank()
long_string_frequency = _build_string_frequency(
    _long_strings,
    case_sensitive=False
)
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

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
def type_2(_string_frequency):

    _character_frequency = {}

    for _string, _ct in _string_frequency.items():
        for _char in str(_string):
            if _char in _character_frequency:
                _character_frequency[_char] += _ct
            else:
                _character_frequency[_char] = _ct

    _character_frequency = dict((zip(
        map(str, _character_frequency.keys()),
        map(int, _character_frequency.values())
    )))


    return _character_frequency
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


assert type_1(long_string_frequency) == type_2(long_string_frequency)
assert type_1(single_string_frequency) == type_2(single_string_frequency)


out = tmb(
    ('type_1a', type_1, [single_string_frequency], {}),
    ('type_1b', type_1, [long_string_frequency], {}),
    ('type_2a', type_2, [single_string_frequency], {}),
    ('type_2b', type_2, [long_string_frequency], {}),
    number_of_trials=10,
    rest_time=1,
    verbose=True
)



