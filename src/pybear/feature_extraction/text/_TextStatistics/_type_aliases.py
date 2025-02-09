# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import (
    TypedDict,
    TypeAlias
)
from typing_extensions import Required

import numbers



class OverallStatisticsType(TypedDict):

    size: Required[numbers.Integral]
    uniques_count: Required[numbers.Integral]
    average_length: Required[numbers.Real]
    std_length: Required[numbers.Real]
    max_length: Required[numbers.Integral]
    min_length: Required[numbers.Integral]


StartsWithFrequencyType: TypeAlias = dict[str, numbers.Integral]

CharacterFrequencyType: TypeAlias = dict[str, numbers.Integral]

WordFrequencyType: TypeAlias = dict[str, numbers.Integral]

LongestWordsType: TypeAlias = dict[str, numbers.Integral]

ShortestWordsType: TypeAlias = dict[str, numbers.Integral]




