# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence, Optional
from .._type_aliases import OverallStatisticsType

import numpy as np

from .._validation._strings import _val_strings


# not used as of first release


def _build_overall_statistics(
    STRINGS: Sequence[str],
    case_sensitive: Optional[bool] = False
) -> OverallStatisticsType:

    """
    Populate a dictionary with the following statistics for the current
    batch of strings:

    - size

    - uniques_count

    - average_length

    - std_length

    - max_length

    - min_length


    Parameters
    ----------
    STRINGS:
        Sequence[str] - a list-like of strings passed to :methods:
        partial_fit or fit.
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

    _val_strings(STRINGS)

    if not isinstance(case_sensitive, bool):
        raise TypeError(f"'case_sensitive' must be boolean")

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    _LENGTHS = np.fromiter(map(len, STRINGS), dtype=np.uint32)

    overall_statistics = {}

    overall_statistics['size'] = len(STRINGS)

    if case_sensitive:
        overall_statistics['uniques_count'] = len(set(STRINGS))
    else:
        overall_statistics['uniques_count'] = len(set(map(str.upper, STRINGS)))

    overall_statistics['average_length'] = float(np.mean(_LENGTHS))
    overall_statistics['std_length'] = float(np.std(_LENGTHS))
    overall_statistics['max_length'] = int(np.max(_LENGTHS))
    overall_statistics['min_length'] = int(np.min(min(_LENGTHS)))

    del _LENGTHS


    return overall_statistics










