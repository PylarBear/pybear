# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
from ..._shared._type_aliases import XWipContainer

import re



def _sep_lb_finder(
    _X: XWipContainer,
    _join_2D: str,
    _sep: Union[re.Pattern[str], tuple[re.Pattern[str], ...]],
    _line_break: Union[None, re.Pattern[str], tuple[re.Pattern[str], ...]]
) -> list[bool]:

    """
    If sep or line_break coincidentally end with join_2D, then find
    which rows end with a sep or a line_break. _X will always be 1D.

    When justifying (which is always in 1D), if the line ended with a
    sep or line_break, then that stayed on the end of the last word. And
    if that sep or line_break ends with the join_2D character string,
    then TextSplitter will leave a relic '' at the end of the
    corresponding rows. So for the case where sep ends with join_2D or
    line_break ends with join_2D, look at the end of each line and
    if ends with sep or line_break then signify that in this list.
    backfill_sep should never be at the end of a line.

    This module tries hard to only find rows where TextJustifierRegExp
    itself put a sep/lb on the end of line and causes a relic ''. It
    also tries hard NOT to touch other rows that don't end in sep or lb
    but the user entry of join_2D caused the join_2D string to be at the
    end of a line (when X goes back to 2D the line will have '' and the
    end of it and the user did it to themself). This module also tries
    hard to use logic that honors the lack of validation between sep and
    line_break in TJRE. Whereas TJ would preclude sep and linebreak from
    simultaneously ending a line (and perhaps the line also ends with
    join_2D), anything goes in TJRE. This module is intended to be
    identical for TJ and TJRE.


    Parameters
    ----------
    _X:
        XWipContainer - The data that has been justified. Need to find
        places where `join_2D` may incidentally coincided with `sep` or
        `line_break` at the end of a line.
    _join_2D:
        str - the character sequence that joined the tokens in each row
        of the data if the data was originally passed as 2D.
    _sep:
        Union[re.Pattern[str], tuple[re.Pattern[str], ...]] - the
        patterns where TextJustifier(RegExp) may have wrapped a line.
    _line_break:
        Union[None, re.Pattern[str], tuple[re.Pattern[str], ...]] - the
        patterns where TextJustifier(RegExp) forced a line break.


    Returns
    -------
    -
        list[bool] - a 1D boolean list signifying which rows will end up
        with a relic '' in the last position.

    """


    assert isinstance(_X, list)
    assert isinstance(_join_2D, str)
    assert isinstance(_sep, (re.Pattern, tuple))
    assert isinstance(_line_break, (type(None), re.Pattern, tuple))


    # join_2D must be a str. the only way this module can be accessed is
    # if _was_2D in the main transform() is True, which means that X
    # was 2D, which means that join_2D was validated and it must be str.

    _MASK = [False for _ in _X]


    def _endswith_helper(_sep: re.Pattern, _line: str) -> bool:
        """Helper function for extracting patterns/flags and adding $."""
        _new_compile = re.compile(f'{_sep.pattern}$', flags=_sep.flags)
        return re.search(_new_compile, _line) is not None


    for _r_idx, _line in enumerate(_X):

        if re.search(f'{re.escape(_join_2D)}$', _line):

            _a =  isinstance(_sep, re.Pattern) and _endswith_helper(_sep, _line)

            _b = isinstance(_sep, tuple) \
                    and any(map(lambda x: _endswith_helper(x, _line), _sep))

            _c = isinstance(_line_break, re.Pattern) \
                 and _endswith_helper(_line_break, _line)

            _d = isinstance(_line_break, tuple) \
                    and any(map(lambda x: _endswith_helper(x, _line), _line_break))

            if _a or _b or _c or _d:
                _MASK[_r_idx] = True


    return _MASK






