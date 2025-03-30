# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import RegExpReplaceArgsType

import re

from ._regexp_param_conditioner import _regexp_param_conditioner



def _regexp_1D_core(
    _X: list[str],
    _regexp_replace: RegExpReplaceArgsType,
) -> list[str]:

    """
    Search and replace strings in whole or in part in a 1D list-like of
    strings using regular expressions.


    Parameters
    ----------
     _X:
        list[str] - the data or a row of data.
    _regexp_replace:
        RegExpReplaceArgsType - the pattern(s) by which to identify
        strings to be replaced and their replacement(s).


    Return
    ------
    -
        list[str]: the 1D vector with string replacements made.

    """


    assert isinstance(_X, list)
    assert all(map(isinstance, _X, (str for _ in _X)))

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    __regexp_replace = _regexp_param_conditioner(_regexp_replace, _X)

    for _idx in range(len(_X)):

        # if set is empty will skip (False in og list becomes empty set)
        for _tuple in __regexp_replace[_idx]:   # set of tuples (or empty)

            # for str_replace, the tuples are already the exact args for str.replace
            _X[_idx] = re.sub(_tuple[0], _tuple[1], _X[_idx], *_tuple[2:])


    return _X







