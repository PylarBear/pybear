# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import (
    XContainer,
    StrReplaceType,
    StrReplaceArgsType
)

from copy import deepcopy



def _str_param_conditioner(
    _str_replace: StrReplaceType,
    _X: XContainer
) -> list[set[StrReplaceArgsType]]:

    """
    Standardize the 'str_replace' parameters into lists of sets of args
    for str.replace, respectively.


    Parameters
    ----------
    _str_replace:
        StrReplaceType - the str.replace parameters.


    Return
    ------
    -
        list[set[StrReplaceArgsType]] - a list of sets of the str.replace
        parameters for each row in X.


    """


    if _str_replace is None:
        _str = [set() for _ in _X]
    elif _str_replace is False:
        # this is a fail-safe. sr could only be False if X is 2D, sr is
        # a list, and we have sent one of the rows of X and its sr value
        # into here. but in main TR.transform False is explicitly skipped
        # pizza u cant pass a bool to str.replace! this needs to be blocked!
        _str = [set((_str_replace,)) for _ in range(len(_X))]
    elif isinstance(_str_replace, tuple):
        _str = [set((_str_replace,)) for _ in _X]
    elif isinstance(_str_replace, set):
        _str = [_str_replace for _ in _X]
    elif isinstance(_str_replace, list):
        if len(_str_replace) != len(_X):
            raise ValueError(f"'len(str_replace) != len(X)")
        _str = deepcopy(_str_replace)
        for _row_idx, _args in enumerate(_str):
            if _args is False:
                _str[_row_idx] = set()
            elif isinstance(_args, tuple):
                _str[_row_idx] = set((_args,))
            elif isinstance(_args, set):
                pass
            else:
                raise TypeError(f'unexpected type {type(_args)} in str_replace')
    else:
        raise TypeError(f"unexpected type {type(_str_replace)} for str_replace")


    return _str



