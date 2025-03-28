# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence, Optional
from typing_extensions import Union

import numpy as np

import re



def _lookup_string(
    _pattern: Union[str, re.Pattern],
    _uniques: Sequence[str],
    _case_sensitive: Optional[bool] = True
) -> list[str]:

    """
    Use string literals or regular expressions to look for whole string
    matches (not substrings) in the fitted words. 'pattern' can be a
    literal string, regular expression, or re.Pattern object.

    If re.Pattern object is passed, case_sensitive is ignored and the
    fitted words are searched with the Pattern object as given. If string
    is passed (which could be a regular expression), build an re.Pattern
    object and apply flags based on case_sensitive. All searches within
    this module are d done with re.fullmatch.

    When searching with string literals and the case_sensitive parameter
    is True, or when searching with re.Pattern objects and case is not
    ignored, look for an identical match to the given pattern. When
    ignoring case (case_sensitive is False for string literals or
    the re.Pattern object has an IGNORECASE flag), perform the search
    looking for any full-word matches without regard to case. If an
    exact match is not found, return an empty list. If matches are found,
    return a 1D list of the matches in their original form from the
    fitted data.

    This is only available if parameter 'store_uniques' in the main
    TextStatistics module is True. If False, the unique strings that
    have been fitted on the TextStatistics instance are not retained
    therefore cannot be searched, and an empty list is always returned.


    Parameters
    ----------
    _pattern:
        Union[str, re.Pattern] - character sequence, regular expression,
        or re.Pattern object to be looked up against the strings fitted
        on the TextStatistics instance.
    _uniques:
        Sequence[str] - the unique strings found by the TextStatistics
        instance during fitting.
    _case_sensitive:
        Optional[bool], default = True - Ignored if an re.Pattern object
        is passed to 'pattern'. If True, search for the exact pattern in
        the fitted data. If False, ignore the case of words in uniques
        while performing the search.


    Return
    ------
    -
        list[str] - if there are any matches, return the matching
        string(s) from the originally fitted data in a 1D list; if there
        are no matches, return an empty list.


    """

    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    if not isinstance(_pattern, (str, re.Pattern)):
        raise TypeError(
            f"'pattern' must be a string (literal or regex) or a "
            f"re.Pattern object."
        )

    if not isinstance(_case_sensitive, bool):
        raise TypeError(f"'case_sensitive' must be boolean")

    try:
        iter(_uniques)
        if isinstance(_uniques, (str, dict)):
            raise Exception
        if not all(map(isinstance, _uniques, (str for _ in _uniques))):
            raise Exception
    except:
        raise TypeError(
            f"'uniques' must be a list-like sequence of strings."
        )

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    if len(_uniques) == 0:
        return []


    # if re.compile was passed, just use that directly.
    # if user passed a literal string or regex, build re.Pattern from it
    if isinstance(_pattern, re.Pattern):
        _re_pattern = _pattern
    else:
        _re_pattern = re.compile(
            _pattern,
            re.I if not _case_sensitive else 0
        )

    # _pattern and _case_sensitive dont matter after here, use _re_pattern


    def _finder(_x: str) -> bool:
        """Helper function for parallel pattern search."""
        nonlocal _re_pattern
        _hit = re.fullmatch(_re_pattern, _x)
        return (_hit is not None and _hit.span() != (0, 0))


    MASK = np.fromiter(map(_finder, _uniques), dtype=bool)

    del _finder

    if np.any(MASK):
        # convert to list so np.array always takes it, covert to ndarray to
        # apply mask, convert to set to get unique strings, then
        # convert back to list.
        return sorted(list(set(map(str, np.array(list(_uniques))[MASK].tolist()))))
    elif not np.any(MASK):
        return []










