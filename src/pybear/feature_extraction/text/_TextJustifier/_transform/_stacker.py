# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numbers



def _stacker(
    _X: list[str],
    _n_chars: numbers.Integral,
    _sep: set[str],
    _line_break: set[str],
    _backfill_sep: str
) -> list[str]:

    """
    After the original text is split into lines of indivisible chunks of
    text by _splitter() (each line ends with sep or line break, or has
    neither), recompile the text line by line to fill the n_chars per
    row requirement. Observe the _line_break rule that that character
    sequence must end a line and subsequent text starts on a new line
    below. Observe the _backfill_sep rule that when a line that does not
    end with a sep has a line from a row below stacked to it, that the
    _backfill_sep sequence is inserted in between.

    How could a line in the split text have no sep or linebreak at the
    end? Because that was how the line was in the raw text, and _splitter
    did a no-op on it because there were no seps or line breaks in it.


    Parameters
    ----------
    pizza
    _X
    _n_chars
    _sep
    _line_break
    _backfill_sep


    Returns
    -------
    _X:
        list[str] - the data in its final justified state.


    """


    # _splitter has turned every line in _X into an indivisible chunk.
    # each string is immutable.
    # sep and line_break must be sets.


    assert isinstance(_sep, set)
    assert isinstance(_line_break, set)


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
        if any(map(lambda x: _line.endswith(x), _line_break)):
            line_idx += 1
            continue


        # backfill onto short lines, conditional on how the line ends
        _needs_backfill_sep = False
        if not any(map(lambda x: _line.endswith(x), _sep)):
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


    return _X














