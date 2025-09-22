# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import (
    Sequence,
)

import re



def _search_always_list(
    _name:str,
    _ALWAYS_LIST:Sequence[str | re.Pattern[str]],
    _word:str
) -> bool:
    """Search for a full pattern match of `_word` against any of the
    literal strings or re.compile objects in `_ALWAYS_LIST`.

    Determine if there are conflicts between any str/re.compile or
    re.compile/re.compile pairs, meaning, do multiple entries indicate
    a match against `_word`.

    Parameters
    ----------
    _name : str
        The name of the special container, e.g. 'SKIP_ALWAYS'.
    _ALWAYS_LIST : Sequence[str | re.Pattern[str]]
        The special word handle list as passed by the user at init plus
        any words added in situ.
    _word:str
        The word in the text body to be checked against the literal
        strings and re.compile objects in `_ALWAYS_LIST`.

    Returns
    -------
    _is_in : bool
        Whether the word from the text body is an exact match against
        any of the string literals or is a regex fullmatch against any
        of the re.compile objects in the _ALWAYS_LIST.

    """

    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    if _name not in [
        'DELETE_ALWAYS', 'REPLACE_ALWAYS', 'SKIP_ALWAYS', 'SPLIT_ALWAYS'
    ]:
        raise ValueError(f"Got invalid name '{_name}'")
    list(iter(_ALWAYS_LIST))
    if isinstance(_ALWAYS_LIST, (str, dict)):
        raise Exception
    assert isinstance(_word, str)
    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    _LITERALS = []
    _REGEX = []
    for item in _ALWAYS_LIST:
        if isinstance(item, str):
            _LITERALS.append(item)
        elif isinstance(item, re.Pattern):
            _REGEX.append(item)
        else:
            raise TypeError(f"{type(item)} in {_name.upper()}")


    _hit_ctr = 0
    # _word == a literal string
    if len(_LITERALS) > 0:
        _hit_ctr += (_word in _LITERALS)
    # _word has fullmatch against a re.compile object
    if len(_REGEX) > 0:
        _re_matches = list(map(lambda x: re.fullmatch(x, _word) is not None, _REGEX))
        if sum(_re_matches) > 1:
            raise ValueError(
                f"'{_name}': there is a conflict between multiple re.compile "
                f"\nobjects causing multiple matches for word '{_word}'."
            )
        _hit_ctr += any(_re_matches)

    if _hit_ctr == 0:
        return False
    elif _hit_ctr == 1:
        return True
    else:
        raise ValueError(
            f"'{_name}': there is a conflict between string literals and "
            f"\nre.compile objects causing multiple matches for word '{_word}'."
        )



