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
    _sep: Union[re.Pattern[str], tuple[re.Pattern[str], ...]],
    _line_break: Union[None, re.Pattern[str], tuple[re.Pattern[str], ...]],
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

    `_sep` and `_line_break` must have already been processed by
    _param_conditioner, i.e., all literal strings must be converted to
    re.compile and any flags passed as parameters or associated with
    `case_sensitive` must have been put in the compile(s).


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
        Union[re.Pattern[str], tuple[re.Pattern[str], ...]] - the regex
        pattern that indicates to TextJustifier(RegExp) where it is
        allowed to wrap a line.
    _line_break:
        Union[None, re.Pattern[str], tuple[re.Pattern[str], ...]] - the
        regex pattern that indicates to TextJustifier(RegExp) where it
        must force a new line.
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

    assert isinstance(_X, list)
    assert isinstance(_n_chars, numbers.Integral)
    assert isinstance(_sep, (re.Pattern, tuple))
    assert isinstance(_line_break, (type(None), re.Pattern, tuple))
    assert isinstance(_backfill_sep, str)


    # _splitter has turned every line in _X into an indivisible chunk.
    # each string is immutable.

    # condition _sep and _line_break to only look at the end -- -- -- --
    def _precondition_helper(
        _obj: Union[None, re.Pattern[str], tuple[re.Pattern[str], ...]]
    ) -> Union[None, tuple[re.Pattern[str], ...]]:

        if _obj is None:
            return None
        elif isinstance(_obj, re.Pattern):
            _new_obj = re.compile(_obj.pattern + '$', _obj.flags)
            # convert this to a tuple for easy iterating later
            _new_obj = (_new_obj, )
        elif isinstance(_obj, tuple):
            _new_obj = []
            for _compile in _obj:
                _new_obj.append(re.compile(_compile.pattern + '$', _compile.flags))
            _new_obj = tuple(_new_obj)
        else:
            raise Exception

        return _new_obj
    # END _precondition_helper -- -- -- -- -- --

    _new_sep = _precondition_helper(_sep)

    _new_line_break = _precondition_helper(_line_break)

    del _precondition_helper
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
        if _new_line_break is not None:    # must be a tuple
            _hit = False
            for _lb in _new_line_break:
                _match = re.search(_lb, _line)
                # pizza .span() == (0, 0) was here
                if _match is not None and _match.span()[1] != _match.span()[0]:
                    line_idx += 1
                    _hit = True
                    break
            if _hit:
                continue
            del _hit, _match

        # backfill onto short lines, conditional on how the line ends
        # if it ends with a separator, do not use a backfill_sep
        _needs_backfill_sep = False
        _hit = False
        for _s in _new_sep:
            _match = re.search(_s, _line)
            # pizza .span() == (0, 0) was here
            if _match is not None and _match.span()[1] != _match.span()[0]:
                _hit = True
                break
        if not _hit:
            _needs_backfill_sep = True
        del _hit, _match

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


    return _X














