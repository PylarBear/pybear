# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import (
    Sequence,
)

import re

from ......base._check_1D_str_sequence import check_1D_str_sequence



def _val_split_always(
    _SPLIT_ALWAYS: None | dict[str | re.Pattern[str], Sequence[str]]
) -> None:
    """Validate `SPLIT_ALWAYS`.

    Must be None or a non-empty dictionary with strings and/or re.compile
    objects as keys and non-empty sequences of strings as values.

    Parameters
    ----------
    _SPLIT_ALWAYS : None | dict[str | re.Pattern[str], Sequence[str]]
        None or a non-empty dictionary with strings and/or re.compile
        objects as keys and sequences of strings as values. When a key
        in the dictionary is a match against a word in the text, the
        matching word is removed and the corresponding words in the
        sequence are inserted, starting in the position of the original
        word.

    Raises
    ------
    ValueError
    TypeError

    Returns
    -------
    None

    """


    if _SPLIT_ALWAYS is None:
        return

    _err_msg = (
        f"'SPLIT_ALWAYS' must be None or a non-empty dictionary with "
        f"strings or re.compile objects as keys and non-empty sequences "
        f"of strings as values. "
    )

    _addon = ""

    try:
        if not isinstance(_SPLIT_ALWAYS, dict):
            raise Exception
        if len(_SPLIT_ALWAYS) == 0:
            _addon = f"Got empty dictionary."
            raise UnicodeError
        for k, v in _SPLIT_ALWAYS.items():
            if not isinstance(k, (str, re.Pattern)):
                _addon = f"Got bad key."
                raise Exception
            try:
                list(iter(v))
                check_1D_str_sequence(v, require_all_finite=True)
            except:
                _addon = f"\nGot bad sequence of strings for key '{k}'."
                raise Exception
            if len(v) == 0:
                _addon = f"\nGot empty sequence for key '{k}'."
                raise UnicodeError
    except UnicodeError:
        raise ValueError(_err_msg + _addon)
    except Exception as e:
        raise TypeError(_err_msg + _addon)





