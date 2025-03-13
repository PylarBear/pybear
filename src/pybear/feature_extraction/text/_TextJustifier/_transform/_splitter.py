# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _splitter(
    _X: list[str],
    _sep: set[str],
    _line_break: set[str]
) -> list[str]:

    """
    Split the text strings in X on all user-defined line breaks and wrap
    separators, so that each line has no breaks or wraps, or if they do,
    the wraps/breaks are at the very end of the string. For each string
    in X, find the first split location, if any, and keep the left side
    of any split in the original row. Insert the right side of the split
    into the index slot immediately after. Then proceed to that next row
    and repeat the procedure until all lines in X are exhausted.


    Parameters
    ----------
    _X:
        list[str] - the data to have individual rows split on line-break
        amd wrap separator string character sequences defined by the user.
    _sep:
        set[str] - the character string sequence(s) that indicate to
        TextJustifier where it is allowed to wrap a line.
    _line_break:
        set[str] - the character string sequence(s) that indicate to
        TextJustifier where it must force a new line.


    Returns
    -------
    _X:
        list[str] - the data split into strings that are not divisible.

    """


    line_idx = 0
    while True:

        try:
            _line = _X[line_idx]
        except IndexError:
            break
        except Exception as e:
            raise e

        # go thru the seps & line_breaks and look for the first hit in
        # the string.
        hit_dict = {
            k:v for k,v in map(lambda x: (_line.find(x), x),
                _sep | _line_break) if k >= 0
        }
        # if no hits, this is zero-len dict, there are no seps or line breaks
        # in that line
        if len(hit_dict) == 0:
            # rows that do not have seps/line_breaks will not be changed
            # increments the line index, and back to the top
            line_idx += 1
            continue

        _lowest_idx = min(hit_dict)
        _first_lb_s = hit_dict[_lowest_idx]
        _adj_lowest_idx = _lowest_idx + len(_first_lb_s) - 1

        del _lowest_idx, _first_lb_s

        # if the hit is in the last position of the string do nothing
        if len(hit_dict) == 1 and _adj_lowest_idx == len(_line)-1:
            line_idx += 1
            continue

        # if there is a hit...
        # keep the left side in the current row
        # insert the right side in the next index after
        if _adj_lowest_idx >= len(_line) - 1:
            _left = _line
            _right = ''
        else:
            _left = _line[:_adj_lowest_idx + 1]
            _right = _line[_adj_lowest_idx + 1:]

        del _line
        _X[line_idx] = _left
        _X.insert(line_idx+1, _right)
        del _left, _right

        line_idx += 1
        continue


    return _X








