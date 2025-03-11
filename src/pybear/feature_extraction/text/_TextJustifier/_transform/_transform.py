# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union

import numbers

from ._split_helper import _split_helper
from ._linebreak_splitter import _linebreak_splitter



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


    Return
    ------
    _X:
        list[str] - the justified text in python list of strings.


    """


    if isinstance(_sep, str):
        _sep = {_sep, }

    if isinstance(_line_break, str):
        _line_break = {_line_break, }
    if _line_break is None:
        _line_break = set()


    # sep can be a substring of line-break, but not vice-versa.
    # split on 'line-break' first, then on 'sep'

    # loop over the entire data set and split on anything that is a line_break.
    # these user-defined line breaks will be in an 'endswith' position
    # on impacted lines.
    # e.g. if X is ['jibberish', 'split this, on a comma.', 'jibberish']
    # then the returned list will be:
    # ['jibberish', 'split this,', 'on a comma.', 'jibberish'] and the comma
    # at the end of 'split this,' is easily recognized with endswith.
    if any(_line_break):
        _X = _linebreak_splitter(_X, _line_break)


    # loop over the entire data set and split on anything that is a sep
    # or a line_break. this way, we will have a 1D vector where everything
    # in it ends with a sep or a line_break.








    #     NEW_TXT = []
    #     for word_idx in range(len(CLEANED_TEXT[row_idx])):
    #         new_word = CLEANED_TEXT[row_idx][word_idx]
    #         if len(seed) + len(new_word) <= max_line_len:
    #             seed += new_word + ' '
    #         elif len(seed) + len(new_word) > max_line_len:
    #             NEW_TXT.append(seed.strip())
    #             seed = new_word + ' '
    # if len(seed) > 0:
    #     NEW_TXT.append(seed.strip())
    #
    # del max_line_len, seed, new_word


    # CLEANED_TEXT = NEW_TXT
    # del NEW_TXT
















