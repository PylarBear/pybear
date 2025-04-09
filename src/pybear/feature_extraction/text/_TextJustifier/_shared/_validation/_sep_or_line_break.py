# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Literal, Sequence
from typing_extensions import Union

import re



def _val_sep_or_line_break(
    _sep_or_line_break:Union[
        None, str, Sequence[str], re.Pattern[str], Sequence[re.Pattern[str]]
    ],
    _name:Literal['sep', 'line_break'],
    _mode:Literal['str', 'regex']
) -> None:

    """
    Validate `sep` or `line_break` for EITHER TextJustifier OR
    TextJustifierRegExp. `sep` cannot be None, but `line_break` can be.
    That is the only difference for what can be passed to `sep` and
    `line_break`.

    TextJustifier:
    Must be a non-empty string or a non-empty python sequence of
    non-empty strings.

    TextJustifierRegExp:
    Must be a re.compile object that does not blatantly return zero-span
    matches or a non-empty python sequence of such objects. re.Pattern
    objects are only validated to be an instance of re.Pattern and to
    not blatantly return zero-span matches. There is no attempt to assess
    the validity of the expression itself. Any exceptions would be raised
    by re.search.


    Parameters
    ----------
    _sep_or_line_break:
        Union[Sequence[str], re.Pattern[str], Sequence[re.Pattern[str]],
        str, None] -

        sep: Union[str, Sequence[str]] - the pattern(s) that indicate to
        TJ/TJRE where it is allowed to wrap a line if n_chars dictates
        to do so. A new line would be wrapped immediately AFTER the
        given pattern. When passed as a sequence of patterns, TJ/TJRE
        will consider any of those patterns as a place where it can
        wrap a line. If the there are no patterns in the data that
        match the given pattern(s), then there are no wraps. If a 'sep'
        pattern match is in the middle of a text sequence that might
        otherwise be expected to be contiguous, TJ/TJRE will wrap a new
        line after the match indiscriminately if proximity to the
        n_chars limit dictates to do so.

        line_break:
        Union[None, re.Pattern[str], Sequence[re.Pattern[str]]] - Tells
        TJ/TJRE where it must start a new line. A new line will be
        started immediately AFTER the given pattern regardless of the
        number of characters in the line. When passed as a sequence of
        patterns, TJ/TJRE will force a new line immediately AFTER any
        occurrences of the patterns given. If None, do not force any
        line breaks. If the there are no patterns in the data that match
        the given pattern(s), then there are no forced line breaks. If
        a line_break pattern is in the middle of a sequence that might
        otherwise be expected to be contiguous, TJ/TJRE will force a new
        line AFTER the line_break indiscriminately. Cannot pass an empty
        string or a regex pattern that blatantly returns a zero-span
        match. Cannot be an empty sequence.
    _name:
        Literal['sep', 'line_break'] - the name of the parameter being
        validated. Must be 'sep' or 'line_break'.
    _mode:
        Literal['str', 'regex'] - whether validating strings for TJ or
        re.compile objects for TJRE.


    Return
    ------
    -
        None

    """


    # validation -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    if _name not in ['sep', 'line_break']:
        raise ValueError(f"'_name' must be 'sep' or 'line_break'")

    if _mode not in ['str', 'regex']:
        raise ValueError(f"'_mode' must be 'str' or 'regex'")
    # END validation -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    if _name == 'line_break' and _sep_or_line_break is None:
        return


    # HELPER FUNCTION -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    def _can_return_empty_match(
        _pat: Union[str, re.Pattern]
    ) -> bool:

        """
        Helper function to try to identify strings or regex patterns
        that will always return zero-span matches.


        Parameters
        ----------
        _pat:
            Union[str, re.Pattern] - a string or re.compile object
            passed to TJ/TJRE at init.


        """

        nonlocal _mode

        if _mode == 'str':
            return (_pat == '')
        elif _mode == 'regex':
            test_strings = ('', 'x')

            for s in test_strings:
                match = _pat.search(s)
                if match and match.span()[0] == match.span()[1]:
                    return True
            return False
        else:
            raise Exception(f'algorithm failure.')

    # END HELPER FUNCTION -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    if _mode == 'str':
        err_msg = (f"'{_name}' must be a non-empty string or a non-empty "
                   f"python sequence of non-empty strings. ")
        if isinstance(_sep_or_line_break, str):
            if _can_return_empty_match(_sep_or_line_break):
                raise ValueError(err_msg + f"Got empty string.")
            return
    elif _mode == 'regex':
        err_msg = (f"'{_name}' must be a re.compile object or python "
                   f"sequence of re.compile objects. \nNo regex patterns "
                   f"that blatantly return zero-span matches are allowed. ")
        if isinstance(_sep_or_line_break, re.Pattern):
            if _can_return_empty_match(_sep_or_line_break):
                raise ValueError(err_msg + f"Got zero-span pattern.")
            return
    else:
        raise Exception(f"algorithm failure.")


    # can only get here if not str/re.compile
    try:
        iter(_sep_or_line_break)
        if isinstance(_sep_or_line_break, (str, dict)):
            raise Exception
    except Exception as e:
        raise TypeError(err_msg + f"Got {_sep_or_line_break}.")

    if len(_sep_or_line_break) == 0:
        raise ValueError(err_msg + f"Got empty sequence.")
    for _item in _sep_or_line_break:
        if _mode == 'str':
            if not isinstance(_item, str):
                raise TypeError(err_msg + f"Got {_item}.")
            if _can_return_empty_match(_item):
                raise ValueError(err_msg + f"Got empty string.")
        elif _mode == 'regex':
            if not isinstance(_item, re.Pattern):
                raise TypeError(err_msg + f"Got {_item}.")
            if _can_return_empty_match(_item):
                raise ValueError(err_msg + f"Got zero-span pattern.")





