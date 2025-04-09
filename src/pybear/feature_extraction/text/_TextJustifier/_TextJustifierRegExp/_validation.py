# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
from .._shared._shared_type_aliases import XContainer

import numbers
import re

from .._shared._validation._sep_or_line_break import _val_sep_or_line_break
from .._shared._validation._X_join_2D import _val_X_join_2D

from ...__shared._validation._any_integer import _val_any_integer
from ...__shared._validation._any_string import _val_any_string



def _validation(
    _X: XContainer,
    _n_chars: numbers.Integral,
    _sep: Union[str, re.Pattern],
    _sep_flags: Union[numbers.Integral, None],
    _line_break: Union[str, re.Pattern, None],
    _line_break_flags: Union[numbers.Integral, None],
    _backfill_sep: str,
    _join_2D: str
) -> None:

    """
    Validate data and parameters for TextJustifierRegExp. This is a
    centralized hub for validation, the brunt of the work is handled by
    the individual modules. See the docs of the individual modules for
    more details.

    'line_break_flags' cannot be passed if 'line_break' is not passed.


    Parameters
    ----------
    _X:
        XContainer - the text to be justified. 2D containers can be
        ragged.
    _n_chars:
        numbers.Integral - the number of characters per line to target
        when justifying the text.
    _sep:
        Union[str, re.Pattern] - the character string sequence(s) that
        indicate to TextJustifierRegExp where it is allowed to wrap a
        line.
    _sep_flags:
        Union[numbers.Integral, None] - the flags for the 'sep' parameter.
    _line_break:
        Union[str, re.Pattern, None] - the character string sequence(s)
        that indicate to TextJustifierRegExp where it must force a new
        line.
    _line_break_flags:
        Union[numbers.Integral, None] - the flags for the 'line_break'
        parameter.
    _backfill_sep:
        str - Some lines in the text may not have any of the given wrap
        separators or line breaks at the end of the line. When justifying
        text and there is a shortfall of characters in a line, TJRE will
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


    _val_X_join_2D(_X, _join_2D)

    if not isinstance(_n_chars, numbers.Integral):
        raise TypeError(f"'n_chars' must be an integer greater than zero.")

    _val_sep_or_line_break(_sep, _name='sep', _mode='regex', _can_be_None=False)

    _val_any_integer(_line_break_flags, 'sep_flags', _can_be_None=True)
    if not isinstance(_line_break_flags, (type(None), numbers.Integral)):
        raise TypeError(f"'sep_flags' must be an integer or None.")

    _val_sep_or_line_break(
        _line_break, _name='line_break', _mode='regex', _can_be_None=True
    )

    _val_any_integer(_line_break_flags, 'line_break_flags', _can_be_None=True)
    if not isinstance(_line_break_flags, (type(None), numbers.Integral)):
        raise TypeError(f"'line_break_flags' must be an integer or None.")

    if _line_break is None and _line_break_flags is not None:
        raise ValueError(
            f"cannot pass 'line_break_flags' when 'line_break' is not passed."
        )

    _val_any_string(_backfill_sep, 'backfill_sep', _can_be_None=False)










