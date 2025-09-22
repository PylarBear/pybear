# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import (
    Sequence,
)

import re



def _val_skip_always(
    _SKIP_ALWAYS: None | Sequence[str | re.Pattern[str]]
) -> None:
    """Validate `SKIP_ALWAYS`.

    Must be None or a 1D sequence of strings and/or re.compile objects.
    Cannot be empty, cannot have duplicate entries.

    Parameters
    ----------
    _SKIP_ALWAYS : None | Sequence[str | re.Pattern[str]]
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


    if _SKIP_ALWAYS is None:
        return


    # check_1D_str_sequence(_SKIP_ALWAYS, require_all_finite=True)

    _err_msg = (f"'SKIP_ALWAYS' must be None or a non-empty 1D list-like "
                f"of strings or re.compile objects. ")

    _addon = ""

    try:
        list(iter(_SKIP_ALWAYS))
        if isinstance(_SKIP_ALWAYS, (str, dict)):
            raise Exception
        if len(_SKIP_ALWAYS) == 0:
            _addon = "Got empty."
            raise UnicodeError
        if not all(map(
            isinstance,
            _SKIP_ALWAYS,
            ((str, re.Pattern) for i in _SKIP_ALWAYS)
        )):
            raise Exception
        if len(_SKIP_ALWAYS) != len(set(_SKIP_ALWAYS)):
            _addon = "Got duplicate entries."
            raise UnicodeError
    except UnicodeError:
        raise ValueError(_err_msg + _addon)
    except Exception as e:
        raise TypeError(_err_msg + _addon)




