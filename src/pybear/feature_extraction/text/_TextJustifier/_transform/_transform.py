# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union

import numbers

from ._splitter import _splitter
from ._stacker import _stacker



def _transform(
    _X: list[str],
    _n_chars: numbers.Integral,
    _sep: Union[str, set[str]],
    _line_break: Union[str, set[str]],
    _backfill_sep: str
) -> list[str]:

    """
    Fit text as strings to user-specified number of characters per row.
    For this module, the data must be a 1D python list of strings.


    Parameters
    ----------
    _X:
        list[str] - The data to justify as a 1D python list of strings.
    _n_chars:
        numbers.Integral - the number of characters per line.
    _sep:
        Union[str, set[str]] - the character string sequence(s) that
        indicate to TextJustifier where it is allowed to wrap a line.
        When passed as a set of strings, TextJustifier will consider any
        of those strings as a place where it can wrap a line; cannot be
        empty.
    _line_break:
        Union[str, set[str], None]] - When passed as a single string,
        TextJustifier will start a new line immediately AFTER all
        occurrences of the character string sequence. When passed as a
        set of strings, TextJustifier will start a new line immediately
        after all occurrences of the character strings given; cannot be
        empty. If None, do not force any line breaks. If the there are
        no string sequences in the data that match the given strings,
        then there are no forced line breaks.
    _backfill_sep:
        Optional[str], default=' ' - when justifying text and there is a
        shortfall of characters in a line, TJ will look to the next line
        to backfill strings. In that case, this character string will
        divide the text from the two lines.


    Return
    ------
    _X:
        list[str] - the justified text in python list of strings.


    """


    if isinstance(_sep, str):
        _sep = {_sep, }

    if _line_break is None:
        _line_break = set()
    elif isinstance(_line_break, str):
        _line_break = {_line_break, }


    # sep can be a substring of line-break, but not vice-versa.
    # split on 'line-break' first, then on 'sep'

    # loop over the entire data set and split on anything that is a line_break
    # or sep. these user-defined line seps/breaks will be in an 'endswith' position
    # on impacted lines.
    # e.g. if X is ['jibberish', 'split this, on a comma.', 'jibberish']
    # then the returned list will be:
    # ['jibberish', 'split this,', 'on a comma.', 'jibberish'] and the comma
    # at the end of 'split this,' is easily recognized with endswith.
    # there must be at least one sep/line_break
    _X = _splitter(_X, _sep, _line_break)


    # we now have a 1D list (still) that has any rows with seps/breaks
    # broken out into indivisible strings on each row.

    # now we need to restack these indivisible units to fill the n_char
    # requirement.
    _X = _stacker(_X, _n_chars, _sep, _line_break, _backfill_sep)


    return _X









