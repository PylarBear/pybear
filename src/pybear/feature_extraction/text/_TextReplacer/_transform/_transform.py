# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import (
    XContainer,
    TRStrReplaceArgsType,
    TRRegExpReplaceArgsType
)

from ._str_1D_core import _str_1D_core
from ._str_2D_core import _str_2D_core
from ._regexp_1D_core import _regexp_1D_core
from ._regexp_2D_core import _regexp_2D_core



def _transform(
    _X: XContainer,
    _str_replace: TRStrReplaceArgsType,
    _regexp_replace: TRRegExpReplaceArgsType
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
        TRStrReplaceArgsType - the search and replace conditions for
        string mode.
    _regexp_replace:
        TRRegExpReplaceArgsType - the search and replace conditions for
        regexp mode.


    Return
    ------
    -
        XContainer: the data with replacements made.


    """


    assert isinstance(_X, list)




    if all(map(isinstance, _X, (str for _ in _X))):

        if _regexp_replace is not None:
            _X = _regexp_1D_core(_X, _regexp_replace)
        if _str_replace is not None:
            _X = _str_1D_core(_X, _str_replace)

    elif all(map(isinstance, _X, (list for _ in _X))):

        raise Exception(f'pizza shouldnt be going into this oven!')

        if _regexp_replace is not None:
            _X = _regexp_2D_core(_X, _regexp_replace)
        if _str_replace is not None:
            _X = _str_2D_core(_X, _str_replace)

    else:
        raise Exception


    del _str_replace, _regexp_replace


    return _X






