# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence
from typing_extensions import Union

import re
import numbers

from ._X import _val_X
from ._sep import _val_sep
from ._regexp import _val_regexp
from ._maxsplit import _val_maxsplit
from ._flags import _val_flags



def _validation(
    _X: Sequence[str],
    _regexp: Union[str, re.Pattern, Sequence[Union[str, re.Pattern]], None],
    _sep: Union[None, str, set[str], list[Union[None, str, set[str]]]],
    _maxsplit: Union[numbers.Integral, Sequence[numbers.Integral], None],
    _flags: Union[numbers.Integral, Sequence[numbers.Integral], None]
) -> None:

    """
    Centralized hub for validating the parameters of TextSplitter.
    See the individual modules for more information.

    
    Parameters
    ----------
    _X:
        Sequence[str] - the data.
    _regexp:
        Union[str, re.Pattern, Sequence[Union[str, re.Pattern]], None] -
        if using regular expressions, the regular expression(s) or the
        re.Pattern object(s).
    _sep:
        Union[None, str, set[str], list[Union[None, str, set[str]]]] -
        if using string.split(), the separator(s) to split on.
    _maxsplit:
        Union[numbers.Integral, Sequence[numbers.Integral], None] - if
        using string.split() or re.split(), the maximum number of splits
        to perform.
    _flags:
        Union[numbers.Integral, Sequence[numbers.Integral], None] - if
        using re.split(), the flags values.


    Return
    ------
    -
        None
     
    """


    _val_X(_X)

    _val_regexp(_regexp, _X)

    _val_sep(_sep, _X)

    _val_maxsplit(_maxsplit, _X)

    _val_flags(_flags, _X)

    # handle the logic for using regexp or str.split().
    # _sep and _regexp cannot simultaneously be entered.
    # if _sep is None, assume the user wants to use str.split(), as
    # it can take a None argument but re.split() cannot. if _sep is None
    # and _regexp is None, _maxsplit must be passed.

    if _sep is not None:
        if _maxsplit is None:
            raise ValueError(f"if passing 'sep', 'maxsplit' must also be passed")
        if _regexp is not None:
            raise ValueError(f"if passing 'sep', 'regexp' cannot be passed")

    if _sep is None:
        if _regexp is None:
            if _maxsplit is None:
                raise ValueError(
                    f"if not passing 'regexp', TextSplitter assumes you are "
                    f"using str.split() and you must pass 'maxsplit'."
                )
        elif _regexp is not None:
            if _maxsplit is not None:
                raise ValueError(
                    f"if passing 'regexp', do not pass 'maxsplit'."
                )








