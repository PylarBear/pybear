# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import (
    XContainer,
    RegExpReplaceType
)

import re



def _regexp_2D_core(
    _X: XContainer,
    _new_regexp_replace: RegExpReplaceType,
) -> XContainer:

    """
    Search and replace strings in whole or in part in a (possibly
    ragged) 2D array-like of strings using regular expressions.


    Parameters
    ----------
     _X:
        XContainer - the data.
    _new_regexp_replace:
        RegExpReplaceType - the pattern(s) by which to identify strings
        to be replaced and their replacement(s).


    Return
    ------
    -
        Union[list[str], list[list[str]]]: the data with string
        replacements made.

    """


    assert isinstance(_X, list)
    assert all(map(isinstance, _X, (list for _ in _X)))

    assert isinstance(_new_regexp_replace, list)
    assert all(map(isinstance, _new_regexp_replace, (set for _ in _X)))

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    for _row_idx in range(len(_X)):

        # if set is empty will skip (False in og list becomes empty set)
        for _tuple in _new_regexp_replace[_row_idx]:  # set of tuples (or empty)

            for _str_idx, _str in enumerate(_X[_row_idx]):

                _X[_row_idx][_str_idx] = \
                    re.sub(_tuple[0], _tuple[1], _X[_row_idx][_str_idx], *_tuple[2:])


    return _X







