# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence

import numpy as np

from ._X import _val_X
from ._str_sep import _val_str_sep
from ._str_maxsplit import _val_str_maxsplit
from ._regexp_sep import _val_regexp_sep
from ._regexp_maxsplit import _val_regexp_maxsplit
from ._regexp_flags import _val_regexp_flags

from .._type_aliases import (
    StrSepType,
    StrMaxSplitType,
    RegExpSepType,
    RegExpMaxSplitType,
    RegExpFlagsType
)



def _validation(
    _X: Sequence[str],
    _str_sep: StrSepType,
    _str_maxsplit: StrMaxSplitType,
    _regexp_sep: RegExpSepType,
    _regexp_maxsplit: RegExpMaxSplitType,
    _regexp_flags: RegExpFlagsType
) -> None:

    """
    Centralized hub for validating the parameters of TextSplitter.
    See the individual modules for more information.
    str.split() parameters and re.split() parameters cannot be passed
    simultaneously.
    If lists are passed to multiple parameters within the same mode,
    all of the Falses in the lists must match.


    
    Parameters
    ----------
    _X:
        Sequence[str] - the data.
    _str_sep:
        StrSepType - if using str.split(), the separator(s) to split
        on.
    _str_maxsplit:
        StrMaxSplitType - if using str.split(), the maximum number of
        splits to perform working left to right.
    _regexp_sep:
        RegExpSepType - if using re.split(), the regular expression(s)
        and/or the re.Pattern object(s) to split with.
    _regexp_maxsplit:
        RexExpMaxSplitType - if using re.split(), the maximum number of
        splits to perform working left to right.
    _regexp_flags:
        RegExpFlagsType - if using re.split(), the flag value(s).


    Return
    ------
    -
        None
     
    """


    _val_X(_X)

    _val_str_sep(_str_sep, _X)

    _val_str_maxsplit(_str_maxsplit, _X)

    _val_regexp_sep(_regexp_sep, _X)

    _val_regexp_maxsplit(_regexp_maxsplit, _X)

    _val_regexp_flags(_regexp_flags, _X)

    # handle the logic for using regexp or str.split().
    # _str and _regexp params cannot be entered simultaneously.

    # if all entries are None (the default), assume the user wants to
    # the default split for str.split(), as it can take a None argument
    # but re.split() cannot.
    # for str:
    # --- if maxsplit is None, use str.split() default
    # for regexp:
    # --- if maxsplit is None, use re.split() default
    # --- if flags is None, use re.split() default


    err_msg = f"cannot simultaneously pass 'str' and 'regexp' parameters"
    if any((_str_sep, _str_maxsplit)) \
            and any((_regexp_sep, _regexp_maxsplit, _regexp_flags)):
        raise ValueError(err_msg)


    if any((_regexp_maxsplit, _regexp_flags)) and _regexp_sep is None:
        raise ValueError(
            f"if passing regexp_maxsplit or regexp_flags, regexp_sep must "
            f"be passed."
        )


    # len of lists passed to params are validated against X in the
    # validation for the individual params. validate that when multiple
    # lists are passed with Falses, the Falses line up.


    err_msg = (
        f"when passing lists to multiple parameters the Falses must "
        f"match against each other. \nthat is, if you are saying that "
        f"splitting is turned off for a string in X (the entry is False "
        f"in one of the lists for that slot), then the corresponding "
        f"slot in all the other passed lists must also be False. "
    )

    _a = isinstance(_str_sep, list)
    _b = isinstance(_str_maxsplit, list)

    _c = isinstance(_regexp_sep, list)
    _d = isinstance(_regexp_maxsplit, list)
    _e = isinstance(_regexp_flags, list)

    if _a and _b:
        if not np.array_equal(
            list(map(lambda x: x is False, _str_sep)),
            list(map(lambda x: x is False, _str_maxsplit))
        ):
            raise ValueError(err_msg)


    if sum((_c, _d, _e)) >= 2:

        if _c and _d:
            if not np.array_equal(
                    list(map(lambda x: x is False, _regexp_sep)),
                    list(map(lambda x: x is False, _regexp_maxsplit))
            ):
                raise ValueError(err_msg)

        if _c and _e:
            if not np.array_equal(
                    list(map(lambda x: x is False, _regexp_sep)),
                    list(map(lambda x: x is False, _regexp_flags))
            ):
                raise ValueError(err_msg)

        if _d and _e:
            if not np.array_equal(
                    list(map(lambda x: x is False, _regexp_maxsplit)),
                    list(map(lambda x: x is False, _regexp_flags))
            ):
                raise ValueError(err_msg)


    del _a, _b, _c, _d, _e












