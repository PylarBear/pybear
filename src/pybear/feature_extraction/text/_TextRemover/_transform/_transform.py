# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import (
    XContainer,
    StrRemoveType,
    RegExpRemoveType,
    RegExpFlagsType,
    RowSupportType
)

import re
import numbers

from ._str_1D_core import _str_1D_core
from ._str_2D_core import _str_2D_core
from ._regexp_1D_core import _regexp_1D_core
from ._regexp_2D_core import _regexp_2D_core



def _transform(
    _X: XContainer,
    _str_remove: StrRemoveType,
    _regexp_remove: RegExpRemoveType,
    _regexp_flags: RegExpFlagsType
) -> tuple[XContainer, RowSupportType]:

    """
    This is a router that sends the data and kwargs to the correct core
    module based on the shape of X and the kwargs.


    Parameters
    ----------
     _X:
        XContainer - the data
    _str_remove:
        StrRemoveType - if in string mode, the strings to match against
        full strings in the data and remove.
    _regexp_remove:
        RegExpRemoveType - if in regexp mode, the string patterns to
        match against full strings in the data and remove.
    _regexp_flags:
        RegExpFlagsType - if in regexp mode, the flags for the string
        patterns.


    Return
    ------
    _
        tuple[XContainer, RowSupportType] - the data with unwanted
        strings removed and the boolean row support vector that indicates
        which rows of the data were kept.


    """


    # validation -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    assert isinstance(_X, list)
    assert all(map(isinstance, _X, ((list, str) for _ in _X)))
    assert isinstance(_str_remove, (type(None), str, set, list))
    assert isinstance(_regexp_remove, (type(None), str, re.Pattern, list))
    assert isinstance(_regexp_flags, (type(None), numbers.Integral, list))
    # END validation -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    if _str_remove is not None:
        if all(map(isinstance, _X, (str for _ in _X))):
            _X, _row_support = _str_1D_core(
                _X,
                _str_remove
            )
        elif all(map(isinstance, _X, (list for _ in _X))):
            _X, _row_support = _str_2D_core(
                _X,
                _str_remove
            )
        else:
            raise Exception
    elif _regexp_remove is not None:
        if all(map(isinstance, _X, (str for _ in _X))):
            _X, _row_support = _regexp_1D_core(
                _X,
                _regexp_remove,
                _regexp_flags
            )
        elif all(map(isinstance, _X, (list for _ in _X))):
            _X, _row_support = _regexp_2D_core(
                _X,
                _regexp_remove,
                _regexp_flags
            )
        else:
            raise Exception
    else:
        raise Exception


    return _X, _row_support






