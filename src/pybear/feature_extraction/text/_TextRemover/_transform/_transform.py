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

from ._str_1D_core import _str_1D_core
from ._str_2D_core import _str_2D_core
from ._regexp_1D_core import _regexp_1D_core
from ._regexp_2D_core import _regexp_2D_core



def _transform(
    _X: XContainer,
    _str_remove: StrRemoveType,
    _regexp_remove: RegExpRemoveType,
    _regexp_flags: RegExpFlagsType
) -> XContainer:


    if _str_remove is not None:
        if isinstance(_X[0], str):
            _X = _str_1D_core(
                _X,
                _str_remove
            )
        elif isinstance(_X[0], list):
            _X = _str_2D_core(
                _X,
                _str_remove
            )
        else:
            raise Exception
    elif _regexp_remove is not None:
        if isinstance(_X[0], str):
            _X = _regexp_1D_core(
                _X,
                _regexp_remove,
                _regexp_flags
            )
        elif isinstance(_X[0], list):
            _X = _regexp_2D_core(
                _X,
                _regexp_remove,
                _regexp_flags
            )
        else:
            raise Exception
    else:
        raise Exception



    return _X






