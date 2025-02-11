# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy as np

from _read_word_bank import _read_word_bank

from pybear.utilities import time_memory_benchmark as tmb



# LINUX
# type_1a     time = 0.254 +/- 0.006 sec; mem = 0.000 +/- 0.000 MB
# type_1b     time = 0.005 +/- 0.001 sec; mem = 0.000 +/- 0.000 MB
# type_1c     time = 0.394 +/- 0.011 sec; mem = 0.000 +/- 0.000 MB
# type_1d     time = 0.021 +/- 0.001 sec; mem = 0.000 +/- 0.000 MB
# type_2a     time = 0.640 +/- 0.011 sec; mem = 0.375 +/- 0.484 MB
# type_2b     time = 0.093 +/- 0.002 sec; mem = 0.000 +/- 0.000 MB
# type_2c     time = 0.723 +/- 0.003 sec; mem = 4.125 +/- 0.599 MB
# type_2d     time = 0.097 +/- 0.001 sec; mem = -3.000 +/- 0.000 MB



# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
_long_strings = _read_word_bank()
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
_raw_strings = _read_word_bank()
_single_strings = []
for line in _raw_strings:
    _single_strings += (line.split(sep=' '))
del _raw_strings
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --



# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
def type_1(STRINGS, case_sensitive):

    _string_frequency = {}
    if case_sensitive:
        for _string in STRINGS:
            _string_frequency[str(_string)] = \
                _string_frequency.get(str(_string), 0) + 1
    elif not case_sensitive:
        for _string in STRINGS:
            _string_frequency[str(_string).upper()] = \
                _string_frequency.get(str(_string).upper(), 0) + 1

    # alphabetize
    for k in sorted(_string_frequency.keys()):
        _string_frequency[str(k)] = _string_frequency.pop(str(k))


    return _string_frequency

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
def type_2(STRINGS, case_sensitive):

    if case_sensitive:
        _string_frequency = dict((zip(*np.unique(STRINGS, return_counts=True))))
    elif not case_sensitive:
        _string_frequency = dict((zip(
            *np.unique(list(map(str.upper, STRINGS)), return_counts=True)
        )))

    _string_frequency = dict((zip(
        map(str, _string_frequency.keys()),
        map(int, _string_frequency.values())
    )))

    return _string_frequency
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


assert type_1(_long_strings, False) == type_2(_long_strings, False)
assert type_1(_long_strings, True) == type_2(_long_strings, True)


out = tmb(
    ('type_1a', type_1, [_single_strings, True], {}),
    ('type_1b', type_1, [_long_strings, True], {}),
    ('type_1c', type_1, [_single_strings, False], {}),
    ('type_1d', type_1, [_long_strings, False], {}),
    ('type_2a', type_2, [_single_strings, True], {}),
    ('type_2b', type_2, [_long_strings, True], {}),
    ('type_2c', type_2, [_single_strings, False], {}),
    ('type_2d', type_2, [_long_strings, False], {}),
    number_of_trials=10,
    rest_time=1,
    verbose=True
)



