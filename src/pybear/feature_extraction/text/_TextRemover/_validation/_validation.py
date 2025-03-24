# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import (
    XContainer,
    StrRemoveType,
    RegExpRemoveType,
    RegExpFlagsType
)

import numpy as np

from ._str_remove import _val_str_remove
from ._regexp_remove import _val_regexp_remove
from ._regexp_flags import _val_regexp_flags

from .....base._check_dtype import check_dtype



def _validation(
    _X: XContainer,
    _str_remove: StrRemoveType,
    _regexp_remove: RegExpRemoveType,
    _regexp_flags: RegExpFlagsType
) -> None:


    """
    Centralized hub for validation. See the individual modules for more
    details.


    Parameters:
    -----------
    _X:
        XContainer - the data.
    _str_remove:
        StrRemoveType - if using str mode, the strings to look for and
        remove.
    _regexp_remove:
        RegExpRemoveType - if using regexp mode, the re patterns to look
        for and remove.
    _regexp_flags:
        RegExpFlagsType = if using regexp mode, the flags objects for the
        re pattern.


    Return
    ------
    -
        None


    """



    try:
        check_dtype(list(_X), allowed='str', require_all_finite=False)
    except Exception as e:
        try:
            check_dtype(list(map(list, _X)), allowed='str', require_all_finite=False)
        except Exception as f:
            raise TypeError(
                f"Expected a 1D sequence or (possibly ragged) 2D array "
                f"of string-like values."
            )

    _val_str_remove(_str_remove, _X)

    _val_regexp_remove(_regexp_remove, _X)

    _val_regexp_flags(_regexp_flags, _X)



    _a = _str_remove is not None
    _b = _regexp_remove is not None
    _c = _regexp_flags is not None


    if not any((_a, _b, _c)):
        raise ValueError(
            f"removal criteria must be entered for either 'str_remove' "
            f"or 'regexp_remove'. cannot leave all the arguments at the "
            f"default value."
        )

    if _a and any((_b, _c)):
        raise ValueError(
            f"cannot pass values for 'str' and 'regexp' parameters "
            f"simultaneously. pass 'str_remove' only, or pass 'regexp_remove' "
            f"and 'regexp_flags' only."
        )

    if _c and not _b:
        raise ValueError(
            f"if trying to use regexp mode, you must pass a pattern to "
            f"'regexp_remove'. cannot pass values to 'regexp_flags' only."
        )

    del _a, _b, _c

    # if 'regexp_remove' and 'regexp_flags' are simultaneously passed as
    # lists and have Falses in them, the Falses must match up.

    err_msg = (
        f"when passing lists to multiple parameters, the Falses must "
        f"match against each other. \nthat is, if you are saying that "
        f"remove is turned off for a string in X (the entry is False "
        f"in one of the lists for that slot), then the corresponding "
        f"slot in all the other passed lists must also be False. "
    )


    if isinstance(_regexp_remove, list) and isinstance(_regexp_flags, list):
        if not np.array_equal(
            list(map(lambda x: x is False, _regexp_remove)),
            list(map(lambda x: x is False, _regexp_flags))
        ):
            raise ValueError(err_msg)







