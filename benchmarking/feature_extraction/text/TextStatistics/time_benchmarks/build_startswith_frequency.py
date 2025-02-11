# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy as np

from _read_word_bank import _read_word_bank

from pybear.feature_extraction.text._TextStatistics._partial_fit. \
    _build_string_frequency import _build_string_frequency

from pybear.utilities import time_memory_benchmark as tmb



# LINUX
# type_1a     time = 0.018 +/- 0.002 sec; mem = 0.000 +/- 0.000 MB
# type_1b     time = 0.004 +/- 0.001 sec; mem = 0.000 +/- 0.000 MB
# type_2a     time = 0.299 +/- 0.017 sec; mem = 0.000 +/- 0.000 MB
# type_2b     time = 0.003 +/- 0.000 sec; mem = 0.000 +/- 0.000 MB
# type_1 wins, simple for loop



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

    _startswith_frequency = {}

    for _string, _ct in _string_frequency.items():
        _startswith_frequency[str(_string[0])] = \
            int(_startswith_frequency.get(str(_string[0]), 0) + _ct)

    return _startswith_frequency
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
def type_2(_string):

    _startswith_frequency = dict((zip(
        *np.unique(np.fromiter(map(lambda x: str(x[0]), _string), dtype='<U1'),
        return_counts=True)
    )))

    _startswith_frequency = dict((zip(
        map(str, _startswith_frequency.keys()),
        map(int, _startswith_frequency.values())
    )))

    return _startswith_frequency
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


assert type_1(long_string_frequency) == type_2(_long_strings)
assert type_1(single_string_frequency) == type_2(_single_strings)


out = tmb(
    ('type_1a', type_1, [single_string_frequency], {}),
    ('type_1b', type_1, [long_string_frequency], {}),
    ('type_2a', type_2, [_single_strings], {}),
    ('type_2b', type_2, [_long_strings], {}),
    number_of_trials=10,
    rest_time=1,
    verbose=True
)



