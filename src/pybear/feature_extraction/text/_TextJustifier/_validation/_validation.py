# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence
from typing_extensions import Union
from .._type_aliases import XContainer

import numbers


from ._n_chars import _val_n_chars
from ._sep import _val_sep
from ._line_break import _val_line_break
from ._join_2D import _val_join_2D
from ._backfill_sep import _val_backfill_sep

from .....base._check_1D_str_sequence import check_1D_str_sequence
from .....base._check_2D_str_array import check_2D_str_array



def _validation(
    _X: XContainer,
    _n_chars: numbers.Integral,
    _sep: Union[str, set[str]],
    _line_break: Union[str, set[str], None],
    _backfill_sep: str,
    _join_2D: Union[str, Sequence[str]]
) -> None:

    """
    Validate data and parameters for TextJustifier.


    Parameters
    ----------
    _X:
        XContainer - the text to be justified. 2D containers can be
        ragged.
    _n_chars:
        numbers.Integral - the number of characters per line.
    _sep:
        Union[str, set[str]] - for 1D containers of (perhaps long)
        strings, the character string sequence(s) that indicate to
        TextJustifier where it is allowed to wrap a line. When passed as
        a set of strings, TextJustifier will consider any of those
        strings as a place where it can wrap a line; cannot be empty.
        TextJustifier processes all data in 1D form (as list of strings),
        with all data given as 2D converted to 1D.
    _line_break:
        Union[str, set[str], None] - When passed as a single string,
        TextJustifier will start a new line immediately AFTER all
        occurrences of the character string sequence. When passed as a
        set of strings, TextJustifier will start a new line immediately
        after all occurrences of the character strings given; cannot be
        empty. If None, do not force any line breaks. If the there are
        no string sequences in the data that match the given strings,
        then there are no forced line breaks.
    backfill_sep:
        str - when justifying text and there is a shortfall of characters
        in a line, TJ will look to the next line to backfill strings. In
        that case, this character string will divide the text from the
        two lines.
    _join_2D:
        Union[str, Sequence[str]] - for 2D containers of (perhaps token)
        strings, the character string sequence(s) that are used to join
        the strings across rows. If a single string, that value is used
        to join for all rows. If a sequence of strings, then the number
        of strings in the sequence must match the number of rows in the
        data, and each entry in the sequence is applied to the
        corresponding entry in the data.


    Return
    ------
    -
        None


    """


    try:
        check_2D_str_array(_X, require_all_finite=True)
        raise UnicodeError
    except UnicodeError:
        # join_2D is ignored if data is 1D
        # need to get rows of off _X.
        if hasattr(_X, 'shape'):
            _n_rows = _X.shape[0]
        else:
            _n_rows = len(_X)
        _val_join_2D(_join_2D, _n_rows)
    except:
        try:
            check_1D_str_sequence(_X, require_all_finite=True)
        except:
            raise TypeError(
                f"TextJustifier expected a 1D sequence of strings or a "
                f"(possibly ragged) 2D array-like of strings. See the docs "
                f"for clarification of accepted containers."
            )


    _val_n_chars(_n_chars)

    _val_sep(_sep)

    _val_line_break(_line_break)

    _val_backfill_sep(_backfill_sep)

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





