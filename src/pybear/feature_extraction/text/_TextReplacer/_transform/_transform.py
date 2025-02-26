# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import (
    XContainer,
    StrReplaceType,
    RegExpReplaceType
)

from ._param_conditioner import _param_conditioner
from ._str_1D_core import _str_1D_core
from ._str_2D_core import _str_2D_core
from ._regexp_1D_core import _regexp_1D_core
from ._regexp_2D_core import _regexp_2D_core



def _transform(
    _X: XContainer,
    _str_replace: StrReplaceType,
    _regexp_replace: RegExpReplaceType
) -> XContainer:

    """
    This module conditions 'str_replace' and 'regexp_replace' into lists
    of sets and routes the parameters into the correct core module based
    on the shape of X.


    Parameters
    ----------
    _X:
        XContainer - the data whose strings will be searched and may
        be replaced in whole or in part.
    _str_replace:
        StrReplaceType - the search and replace conditions for string
        mode.
    _regexp_replace:
        RegExpReplaceType - the search and replace conditions for regexp
        mode.


    Return
    ------
    -
        XContainer: the data with replacements made.


    """


    assert isinstance(_X, list)


    __str_replace, __regexp_replace = _param_conditioner(
        _str_replace,
        _regexp_replace,
        _X
    )


    if all(map(isinstance, _X, (str for _ in _X))):

        _X = _regexp_1D_core(_X, __regexp_replace)
        _X = _str_1D_core(_X, __str_replace)

    elif all(map(isinstance, _X, (list for _ in _X))):

        _X = _regexp_2D_core(_X, __regexp_replace)
        _X = _str_2D_core(_X, __str_replace)

    else:
        raise Exception


    del __str_replace, __regexp_replace


    return _X






