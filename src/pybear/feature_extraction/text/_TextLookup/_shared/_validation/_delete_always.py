# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import (
    Sequence,
)

import re



def _val_delete_always(
    _DELETE_ALWAYS: None | Sequence[str | re.Pattern[str]]
) -> None:
    """Validate `DELETE_ALWAYS`.

    Must be None or a 1D sequence of strings and/or re.compile objects.
    Cannot be empty, cannot have duplicate entries.

    Parameters
    ----------
    _DELETE_ALWAYS : None | Sequence[str | re.Pattern[str]]
        A non-empty 1D sequence of strings and/or re.compile objects that
        when there is a match against a word in the text, that word is
        removed from the body of text.

    Raises
    ------
    ValueError
    TypeError

    Returns
    -------
    None

    """


    if _DELETE_ALWAYS is None:
        return


    # check_1D_str_sequence(_DELETE_ALWAYS, require_all_finite=True)

    _err_msg = (f"'DELETE_ALWAYS' must be None or a non-empty 1D list-like "
                f"of strings or re.compile objects. ")

    _addon = ""

    try:
        list(iter(_DELETE_ALWAYS))
        if isinstance(_DELETE_ALWAYS, (str, dict)):
            raise Exception
        if len(_DELETE_ALWAYS) == 0:
            _addon = "Got empty."
            raise UnicodeError
        if not all(map(
            isinstance,
            _DELETE_ALWAYS,
            ((str, re.Pattern) for i in _DELETE_ALWAYS)
        )):
            raise Exception
        if len(_DELETE_ALWAYS) != len(set(_DELETE_ALWAYS)):
            _addon = f"Got duplicate entries."
            raise UnicodeError
    except UnicodeError:
        raise ValueError(_err_msg + _addon)
    except Exception as e:
        raise TypeError(_err_msg + _addon)




