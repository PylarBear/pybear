# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence, Optional
from .._type_aliases import OverallStatisticsType

import numpy as np

from .._validation._words import _val_words



def _build_overall_statistics(
    WORDS: Sequence[str],
    case_sensitive: Optional[bool] = False
) -> OverallStatisticsType:

    """
    Populate a dictionary with the following statistics for the current
    batch of words:
    - size
    - uniques_count
    - average_length
    - std_length
    - max_length
    - min_length


    Parameters
    ----------
    WORDS:
        Sequence[str] - a list-like of strings
    case_sensitive:
        Optional[bool], default = False - whether to normalize all
        characters to the same case or preserve the original case.


    Return
    ------
    -
        overall_statistics: dict[str, numbers.Real] - the statistics for
        the current batch of data.


    """


    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    _val_words(WORDS)

    if not isinstance(case_sensitive, bool):
        raise TypeError(f"'case_sensitive' must be boolean")

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    _LENGTHS = np.fromiter(map(len, WORDS), dtype=np.uint32)

    overall_statistics = {}

    overall_statistics['size'] = len(WORDS)

    if case_sensitive:
        overall_statistics['uniques_count'] = len(set(WORDS))
    else:
        overall_statistics['uniques_count'] = len(set(map(str.upper, WORDS)))

    overall_statistics['average_length'] = float(np.mean(_LENGTHS))
    overall_statistics['std_length'] = float(np.std(_LENGTHS))
    overall_statistics['max_length'] = int(np.max(_LENGTHS))
    overall_statistics['min_length'] = int(np.min(min(_LENGTHS)))

    del _LENGTHS


    return overall_statistics










