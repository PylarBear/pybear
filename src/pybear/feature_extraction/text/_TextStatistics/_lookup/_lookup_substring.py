# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence, Optional

import numpy as np



def _lookup_substring(
    char_seq: str,
    uniques: Sequence[str],
    case_sensitive: Optional[bool] = True
) -> list[str]:

    """
    Return a list of all strings that have been fitted on the
    TextStatistics instance that contain the given character substring.
    This is only available if parameter 'store_uniques' is True. If
    False, the unique strings that have been fitted on the TextStatistics
    instance are not retained therefore cannot be searched and an empty
    list is returned.


    Parameters
    ----------
    char_seq:
        str - character substring to be looked up against the strings
        fitted on the TextStatistics instance.
    uniques:
        Sequence[str] - the unique strings found by the TextStatistics
        instance during fitting.
    case_sensitive:
        Optional[bool], default = True - If True, search for the
        exact string in the fitted data. If False, normalize both
        the given string and the strings fitted on the TextStatistics
        instance, then perform the search.


    Return
    ------
    -
        SELECTED_STRINGS: list[str] - list of all strings in the fitted
        data that contain the given character substring. Returns an
        empty list if there are no matches.


    """


    if not isinstance(char_seq, str):
        raise TypeError(f"'char_seq' must be a string")

    if not isinstance(case_sensitive, bool):
        raise TypeError(f"'case_sensitive' must be boolean")

    try:
        iter(uniques)
        if isinstance(uniques, (str, dict)):
            raise Exception
        if not all(map(isinstance, uniques, (str for _ in uniques))):
            raise Exception
    except:
        raise TypeError(
            f"'uniques' must be a list-like sequence of strings."
        )

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *



    if not len(uniques):
        return []


    def _finder(x: str) -> int:
        nonlocal _char_seq
        return x.find(_char_seq) + 1


    if case_sensitive:
        _char_seq = char_seq
        MASK = np.fromiter(map(_finder, uniques), dtype=bool)
    else:
        _char_seq = char_seq.lower()
        MASK = np.fromiter(
            map(_finder, np.char.lower(list(uniques))),
            dtype=bool
        )


    SELECTED_STRINGS = list(map(str, np.array(list(uniques))[MASK]))

    del _finder, MASK


    return SELECTED_STRINGS









