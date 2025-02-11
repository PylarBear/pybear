# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence, Optional
from .._type_aliases import StringFrequencyType

import numpy as np

from .._validation._strings import _val_strings



def _build_string_frequency(
    STRINGS: Sequence[str],
    case_sensitive: Optional[bool] = False
) -> StringFrequencyType:

    """
    Build a dictionary of the unique strings in STRINGS and their counts.


    Parameters
    ----------
    STRINGS:
        Sequence[str] - the sequence of strings currently being fitted.
    case_sensitive:
        Optional[bool], default = False - whether to preserve the case
        of the characters when getting the uniques. When False, normalize
        the case of all characters.


    Return
    ------
    -
        dict[str, int] - a dictionary with the unique strings in STRINGS
        as keys and their respective counts as values.


    """


    _val_strings(STRINGS)

    if not isinstance(case_sensitive, bool):
        raise TypeError(f"'case_sensitive' must be boolean")

    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

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






