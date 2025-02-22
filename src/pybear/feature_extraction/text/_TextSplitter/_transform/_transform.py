# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence
from typing_extensions import Union

import re
import numbers

from .._type_aliases import XContainer

from .._validation._validation import _validation



def _transform(
    _X: XContainer,
    _sep: Union[str, Sequence[str], None],
    _regexp: Union[str, Sequence[str], None],
    _maxsplit: Union[numbers.Integral, Sequence[numbers.Integral], None],
    _flags: Union[numbers.Integral, Sequence[numbers.Integral], None]
) -> list[list[str]]:

    """
    Split all strings in X on the given separator(s).
    
    
    Parameter
    ---------
    _X:
        Sequence[str] - the data.
    _sep:
        _sep: Union[None, str, set[str], list[Union[str, set[str], None]]] -
        the separator(s) to split the text strings on.
    _regexp:
        Union[str, re.Pattern, Sequence[Union[str, re.Pattern]], None] -
        pizza
    _maxsplit:
        Union[numbers.Integral, Sequence[numbers.Integral], None] -
        pizza
    _flags:
        Union[numbers.Integral, Sequence[numbers.Integral], None] -
         pizza


    Return
    ------
    _X:
        2D array-like, may be ragged - the data split on the given
        separators.
    
     
    """



    _validation(_X, _regexp, _sep, _maxsplit, _flags)


    _X = list(_X)


    if _regexp is not None:
        # then definitely using re.split()
        if isinstance(_regexp, (str, re.Pattern)):
            __regexp = [_regexp for _ in _X]
        else:
            # must be sequence
            __regexp = _regexp

        if isinstance(_maxsplit, (numbers.Integral, type(None))):
            __maxsplit = [_maxsplit for _ in _X]
        else:
            # must be sequence
            __maxsplit = _maxsplit

        if isinstance(_flags, numbers.Integral):
            __flags = [_flags for _ in _X]
        else:
            # must be sequence
            __flags = _flags

        for _idx, _str in enumerate(_X):
            _X[_idx] = re.split(__regexp[_idx], _str, __maxsplit[_idx], __flags[_idx])



    elif _regexp is None:
        # then using str.split()

        if isinstance(_maxsplit, numbers.Integral):
            __maxsplit = [_maxsplit for _ in _X]
        else:
            # must be a sequence
            __maxsplit = _maxsplit

        if isinstance(_sep, (str, type(None))):
            __sep = [_sep for _ in _X]
        else:
            # must be a sequence
            __sep = _sep

        for _idx, _str in enumerate(_X):
            if isinstance(__sep[_idx], (str, type(None))):
                _X[_idx] = _str.split(sep=__sep[_idx], maxsplit=__maxsplit[_idx])
            else:
                for _i in __sep[_idx][1:]:
                    _str = _str.split(_i, __sep[_idx][0])

                _X[_idx] = _str.split(sep=__sep[_idx][0], maxsplit=__maxsplit[_idx])


    return _X














