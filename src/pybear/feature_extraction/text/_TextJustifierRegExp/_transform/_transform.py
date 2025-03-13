# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union

import numbers
import re

from ._splitter import _splitter
from ._stacker import _stacker



def _transform(
    _X: list[str],
    _n_chars: numbers.Integral,
    _sep: Union[str, re.Pattern],
    _sep_flags: Union[numbers.Integral, None],
    _line_break: Union[str, re.Pattern, None],
    _line_break_flags: Union[numbers.Integral, None],
    _backfill_sep: str
) -> list[str]:

    """
    Fit text as strings to the user-specified number of characters per
    row. For this module, the data must be a 1D python list of strings.


    Parameters
    ----------
    _X:
        list[str] - The text to justify as a 1D python list of strings.
    _n_chars:
        numbers.Integral - the number of characters per line to target
        when justifying the text.
    _sep:
        Union[str, re.Pattern] - the regexp pattern that indicates to
        TextJustifierRegExp where it is allowed to wrap a line.
    _sep_flags:
        Union[numbers.Integral, None] - the flags for the 'sep' parameter.
    _line_break:
        Union[str, re.Pattern, None]] - the regexp pattern that indicates
        to TextJustifierRegExp where it must force a new line.
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


    Return
    ------
    _X:
        list[str] - the justified text in python list of strings.


    """


    # loop over the entire data set and split on anything that is a line_break
    # or sep. these user-defined line seps/breaks will be in an 'endswith'
    # position on impacted lines.
    # e.g. if X is ['jibberish', 'split this, on a comma.', 'jibberish']
    # then the returned list will be:
    # ['jibberish', 'split this,', 'on a comma.', 'jibberish'] and the comma
    # at the end of 'split this,' is easily recognized with endswith.
    _X = _splitter(_X, _sep, _sep_flags, _line_break, _line_break_flags)


    # we now have a 1D list (still) that has any rows with seps/breaks
    # broken out into indivisible strings on each row.

    # now we need to restack these indivisible units to fill the n_char
    # requirement.
    _X = _stacker(
        _X, _n_chars, _sep, _sep_flags, _line_break, _line_break_flags,
        _backfill_sep
    )


    return _X









