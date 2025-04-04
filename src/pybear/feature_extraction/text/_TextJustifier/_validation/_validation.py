# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
from .._type_aliases import XContainer

import numbers


from ._n_chars import _val_n_chars
from ._sep import _val_sep
from ._line_break import _val_line_break

from ...__shared._validation._1D_X import _val_1D_X
from ...__shared._validation._2D_X import _val_2D_X
from ...__shared._validation._any_string import _val_any_string



def _validation(
    _X: XContainer,
    _n_chars: numbers.Integral,
    _sep: Union[str, set[str]],
    _line_break: Union[str, set[str], None],
    _backfill_sep: str,
    _join_2D: str
) -> None:

    """
    Validate data and parameters for TextJustifier. This is a centralized
    hub for validation, the brunt of the work is handled by the
    individual modules. See the docs of the individual modules for more
    details.

    No seps can be identical and one cannot be a substring of another.
    No sep can be identical to a line_break entry and no sep can be a
    substring of a line_break. No line_breaks can be identical and one
    cannot be a substring of another. No line_break can be identical to
    a sep entry and no line_break can be a substring of a sep.


    Parameters
    ----------
    _X:
        XContainer - the text to be justified. 2D containers can be
        ragged.
    _n_chars:
        numbers.Integral - the number of characters per line to target
        when justifying the text.
    _sep:
        Union[str, set[str]] - the character string sequence(s) that
        indicate to TextJustifier where it is allowed to wrap a line.
    _line_break:
        Union[str, set[str], None] - the character string sequence(s)
        that indicate to TextJustifier where it must force a new line.
    _backfill_sep:
        str - Some lines in the text may not have any of the given wrap
        separators or line breaks at the end of the line. When justifying
        text and there is a shortfall of characters in a line, TJ will
        look to the next line to backfill strings. In the case where the
        line being backfilled onto does not have a separator or line
        break at the end of the string, this character string will
        separate the otherwise separator-less strings from the strings
        being backfilled onto them.
    _join_2D:
        str - Ignored if the data is given as a 1D sequence. For 2D
        containers of strings, this is the character string sequence
        that is used to join the strings across rows. The single string
        value is used to join for all rows.


    Return
    ------
    -
        None


    """


    try:
        _val_2D_X(_X, _require_all_finite=True)
        raise UnicodeError
    except UnicodeError:
        # join_2D is ignored if data is 1D
        _val_any_string(_join_2D, 'join_2D', _can_be_None=False)
    except:
        try:
            _val_1D_X(_X, _require_all_finite=True)
        except:
            raise TypeError(
                f"TextJustifier expected a 1D sequence of strings or a "
                f"(possibly ragged) 2D array-like of strings. See the docs "
                f"for clarification of accepted containers."
            )


    _val_n_chars(_n_chars)

    _val_sep(_sep)

    _val_line_break(_line_break)

    _val_any_string(_backfill_sep, 'backfill_sep', _can_be_None=False)

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    err_msg = (
        f"there is a conflict between strings for 'sep' and 'line_break'. "
        f"\nno 'sep' and 'line_break' character sequences can be identical. "
        f"\nno 'line_break' can be a substring of any 'sep'. "
        f"\nno 'sep' can be a substring of any 'line_break'. "
        f"\nno 'sep' can be a substring of another 'sep'. "
        f"\nno 'line_break' can be a substring of another 'line_break'. "
    )

    if isinstance(_sep, str):
        set1 = {_sep,}
    else:
        set1 = _sep.copy()

    if _line_break is None:
        set2 = set()
    elif isinstance(_line_break, str):
        set2 = {_line_break,}
    else:
        set2 = _line_break.copy()

    _union = set1 | set2

    if len(_union) != len(set1) + len(set2):
        raise ValueError(err_msg)
    # we know there are no exact duplicates
    # now find if there are any shared substrings
    for s1 in _union:
        if any(s1 in s2 for s2 in _union if s2 != s1):
            raise ValueError(err_msg)

    del err_msg, set1, set2, _union





