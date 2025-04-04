# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
from .._type_aliases import XContainer

import numbers
import re

from ._n_chars import _val_n_chars
from ._sep import _val_sep
from ._sep_flags import _val_sep_flags
from ._line_break import _val_line_break
from ._line_break_flags import _val_line_break_flags

from ...__shared._validation._1D_X import _val_1D_X
from ...__shared._validation._2D_X import _val_2D_X
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
    Validate data and parameters for TextJustifierRegExp. This is a centralized
    hub for validation, the brunt of the work is handled by the
    individual modules. See the docs of the individual modules for more
    details.


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
                f"TextJustifierRegExp expected a 1D sequence of strings "
                f"or a (possibly ragged) 2D array-like of strings. \nSee "
                f"the docs for clarification of accepted containers."
            )


    _val_n_chars(_n_chars)

    _val_sep(_sep)

    _val_sep_flags(_sep_flags)

    _val_line_break(_line_break)

    _val_line_break_flags(_line_break_flags)

    if _line_break is None and _line_break_flags is not None:
        raise ValueError(
            f"cannot pass 'line_break_flags' when 'line_break' is not passed."
        )

    _val_any_string(_backfill_sep, 'backfill_sep', _can_be_None=False)










