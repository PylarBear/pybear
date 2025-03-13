# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union

import re



def _val_line_break(
    _line_break: Union[str, re.Pattern, None]
) -> None:

    """
    Validate 'line_break'. Must be None, a regexp pattern or a re.Pattern
    object.


    Parameters
    ----------
    _line_break:
        Union[str, set[str], None] - the regexp pattern that tells TJRE
        where it must start a new line. A new line will be started
        immediately AFTER all occurrences of the pattern regardless of
        the number of characters in the line. If None, do not force any
        line breaks. If the there are no patterns in the data that match,
        then there are no forced line breaks. If a line_break pattern is
        in the middle of a sequence that might otherwise be expected to
        be contiguous, TJRE will force a new line after the line_break
        indiscriminately.


    Return
    ------
    -
        None

    """


    if _line_break is None:
        return


    err_msg = (f"'line_break' must be a regexp string or re.Pattern object. ")


    if not isinstance(_line_break, (str, re.Pattern)):
        raise TypeError(err_msg)






