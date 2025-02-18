# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from _read_word_bank import _read_word_bank

from pybear.feature_extraction.text._TextStatistics._partial_fit. \
    _build_overall_statistics_OLD import _build_overall_statistics


_raw_strings = _read_word_bank()
_single_strings = []
for line in _raw_strings:
    _single_strings += (line.split(sep=' '))
del _raw_strings




out = _build_overall_statistics(_single_strings, case_sensitive=True)

[print(f"{k}: {v}") for k, v in out.items()]


