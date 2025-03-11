# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from ._split_helper import _split_helper



def _sep_splitter(
    _X: list[str],
    _sep: set[str]
) -> list[str]:

    """
    Split the text strings in X on user-defined line separator strings.
    Keep the left side of the split in the original row. Insert the
    right side of the split into the index slot immediately after.


    Parameters
    ----------
    _X:
        list[str] - the data to have individual rows wrapped on separator
        string character sequences defined by the user.
    _sep:
        set[str] - the separator string character sequence(s) defined
        by the user.


    Returns
    -------
    _X:
        list[str] - the data split into new strings

    """


    line_idx = 0
    while True:

        try:
            _line = _X[line_idx]
        except IndexError:
            break
        except Exception as e:
            raise e

        # go thru the line_breaks and look for the first hit
        # if no hits, this skips out, increments the line, and back to the top
        for _lb in _sep:
            _lb_idx = _line.find(_lb)
            if _lb_idx >= 0:
                if _lb_idx == len(_line)-1:
                    continue
                _lb_idx += len(_lb)-1
            else:
                continue

            # if there is a hit...

            # keep the left side in the current row
            # insert the right side after
            _left, _right = _split_helper(_line, _lb_idx)
            del _line, _lb_idx
            _X[line_idx] = _left
            _X.insert(line_idx+1, _right)
            del _left, _right

            # then get out of this for loop
            # and go to the next line (which is the line we just inserted
            # below)
            break


        line_idx += 1
        continue


    return _X








