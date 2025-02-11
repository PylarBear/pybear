# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence, Optional
from typing_extensions import Union

import numpy as np



def _lookup_string(
    char_seq: str,
    uniques: Sequence[str],
    case_sensitive: Optional[bool] = True
) -> Union[str, list[str], None]:

    """
    Look in the fitted strings for a full character sequence (not a
    substring) that exactly matches the given character sequence. If
    the case_sensitive parameter is True, look for an identical match to
    the given character sequence, and if at least one is found, return
    that character string. If an exact match is not found, return None.
    If the case_sensitive parameter is False, normalize the given
    character string and the strings seen by the TextStatistics instance
    and search for matches. If matches are found, return a 1D list
    of the matches in their original form from the fitted data (there
    may be matches with different capitalization in the fitted data, so
    there may be multiple entries.) If no matches are found, return None.


    Parameters
    ----------
    char_seq:
        str - character string to be looked up against the strings
        fitted on the TextStatistics instance.
    uniques:
        Sequence[str] - the unique strings found by the TextStatistics
        instance during fitting.
    case_sensitive:
        Optional[bool], default = True - If True, search for the exact
        string in the fitted data. If False, normalize both the given
        string and the strings fitted on the TextStatistics instance,
        then perform the search.


    Return
    ------
    -
        Union[str, list[str], None] - if there are any matches, return
        the matching string(s) from the originally fitted data; if there
        are no matches, return None.


    """


    if not isinstance(char_seq, str):
        raise TypeError(f"'char_seq' must be a string")

    if not isinstance(case_sensitive, bool):
        raise TypeError(f"'case_sensitive' must be boolean")

    try:
        iter(uniques)
        if isinstance(uniques, (str, dict)):
            raise Exception
        if len(uniques) == 0:
            raise Exception
        if not all(map(isinstance, uniques, (str for _ in uniques))):
            raise Exception
    except:
        raise TypeError(
            f"'uniques' must be a list-like sequence of strings, "
            f"cannot be empty."
        )

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    def _finder(x: str) -> bool:
        nonlocal _char_seq
        return _char_seq == x


    if case_sensitive:
        _char_seq = char_seq
        MASK = np.fromiter(map(_finder, uniques), dtype=bool)
    else:
        _char_seq = char_seq.lower()
        MASK = np.fromiter(
            map(_finder, np.char.lower(list(uniques))),
            dtype=bool
        )

    if case_sensitive:
        if np.any(MASK):
            return char_seq
        elif not np.any(MASK):
            return
    elif not case_sensitive:
        if np.any(MASK):
            # convert to list so np.array takes it, covert to ndarray to
            # apply mask, convert to set to get unique strings, then
            # convert back to list.
            return list(map(str, list(set(np.array(list(uniques))[MASK].tolist()))))
        elif not np.any(MASK):
            return










