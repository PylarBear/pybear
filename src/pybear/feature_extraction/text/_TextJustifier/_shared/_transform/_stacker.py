# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union

import numbers
import re



def _stacker(
    _X: list[str],
    _n_chars: numbers.Integral,
    _sep: Union[str, re.compile],
    _sep_flags: Union[numbers.Integral, None],
    _line_break: Union[str, re.compile, None],
    _line_break_flags: Union[numbers.Integral, None],
    _backfill_sep: str
) -> list[str]:

    """
    After the original text is split into lines of indivisible chunks of
    text by _splitter() (each line ends with sep or line break, or has
    neither), recompile the text line by line to fill the n_chars per
    row requirement. Observe the _line_break rule that that pattern must
    end a line and subsequent text starts on a new line below. Observe
    the _backfill_sep rule that when a line that does not end with a sep
    pattern has a line from a row below stacked to it, that the
    _backfill_sep sequence is inserted in between.

    How could a line in the split text have no sep or linebreak at the
    end? Because that was how the line was in the raw text, and _splitter
    did a no-op on it because there were no seps or line breaks in it.


    Parameters
    ----------
    _X:
        list[str] - the data as processed by _splitter(). Must be a list
        of strings. Each string is an indivisible unit of text based on
        the given separators and line-breaks.
    _n_chars:
        numbers.Integral - the number of characters per line to target
        when justifying the text.
    _sep:
        Union[str, re.Pattern] - the regexp pattern that indicates to
        TextJustifierRegExp where it is allowed to wrap a line.
    _sep_flags:
        Union[numbers.Integral, None] - the flags for the 'sep' parameter.
    _line_break:
        Union[str, re.Pattern, None] - the regexp pattern that indicates
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


    Returns
    -------
    _X:
        list[str] - the data in its final justified state.


    """


    # _splitter has turned every line in _X into an indivisible chunk.
    # each string is immutable.

    _sep_kwargs = {} if _sep_flags is None else {'flags': _sep_flags}

    _line_break_kwargs = \
        {} if _line_break_flags is None else {'flags': _line_break_flags}

    # condition _sep and _line_break to only look at the end -- -- -- --
    if isinstance(_sep, str):
        _new_sep = _sep + '$'
    elif isinstance(_sep, re.Pattern):
        _new_sep = re.compile(_sep.pattern + '$', _sep.flags)
    else:
        raise Exception

    if _line_break is None:
        pass
    elif isinstance(_line_break, str):
        _new_line_break = _line_break + '$'
    elif isinstance(_line_break, re.Pattern):
        _new_line_break = re.compile(_line_break.pattern + '$', _line_break.flags)
    else:
        raise Exception
    # END condition _sep and _line_break to only look at the end -- -- --


    line_idx = 0
    while True:

        _line = _X[line_idx]

        # if the next line doesnt exist we cant do anymore
        try:
            _X[line_idx + 1]
        except IndexError:
            break
        except Exception as e:
            raise e


        # if a line is already at or over n_chars, go to the next line
        if len(_line) >= _n_chars:
            line_idx += 1
            continue

        # all lines below here are shorter than n_chars

        # if the line ends with a line_break, nothing can be backfilled
        # onto it
        if _line_break is not None:
            _match = re.search(_new_line_break, _line, **_line_break_kwargs)
            if _match is not None and _match.span() != (0, 0):
                line_idx += 1
                continue

        # backfill onto short lines, conditional on how the line ends
        _needs_backfill_sep = False
        _match = re.search(_new_sep, _line, **_sep_kwargs)
        if _match is None or _match.span() == (0, 0):
            _needs_backfill_sep = True

        _addon_len = len(_X[line_idx + 1])
        # if the current line has no sep at the end, append backfill_sep
        _addon_len += len(_backfill_sep) if _needs_backfill_sep else 0

        if len(_line) + _addon_len <= _n_chars:
            if _needs_backfill_sep:
                _X[line_idx] += _backfill_sep
            _X[line_idx] += _X.pop(line_idx + 1)
            # do not increment the line index!
            # see if more lines can be pulled up to the current line from below
            continue
        else:
            # attaching addon puts it over _n_chars
            line_idx += 1
            continue


    del _sep_kwargs, _line_break_kwargs


    return _X














