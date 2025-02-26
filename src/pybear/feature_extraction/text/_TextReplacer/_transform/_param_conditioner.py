# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import (
    XContainer,
    StrReplaceType,
    RegExpReplaceType,
    StrReplaceArgsType,
    RegExpReplaceArgsType
)

from copy import deepcopy



def _param_conditioner(
    _str_replace: StrReplaceType,
    _regexp_replace: RegExpReplaceType,
    _X: XContainer
) -> tuple[list[set[StrReplaceArgsType]], list[set[RegExpReplaceArgsType]]]:

    """
    Standardize the 'str_replace' and 'regexp_replace' parameters into
    lists of sets of args for str.replace and re.sub, respectively.


    Parameters
    ----------
    _str_replace:
        StrReplaceType - the str.replace parameters.

    _regexp_replace:
        RegExpReplaceType - the re.sub parameters.


    Return
    ------
    -
        tuple[list[set[StrReplaceArgsType]], list[set[RegExpReplaceArgsType]]] -
        a list of sets of the str.replace parameters for each row in X
        and another list of sets of the re.sub parameters for each row
        in X.


    """


    if _str_replace is None:
        _str = [set() for _ in _X]
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


    if _regexp_replace is None:
        _regexp = [set() for _ in _X]
    elif isinstance(_regexp_replace, tuple):
        _regexp = [set((_regexp_replace,)) for _ in _X]
    elif isinstance(_regexp_replace, set):
        _regexp = [_regexp_replace for _ in _X]
    elif isinstance(_regexp_replace, list):
        if len(_regexp_replace) != len(_X):
            raise ValueError(f"'len(regexp_replace) != len(X)")
        _regexp = deepcopy(_regexp_replace)
        for _row_idx, _args in enumerate(_regexp):
            if _args is False:
                _regexp[_row_idx] = set()
            elif isinstance(_args, tuple):
                _regexp[_row_idx] = set((_args,))
            elif isinstance(_args, set):
                pass
            else:
                raise TypeError(f'unexpected type {type(_args)} in regexp_replace')
    else:
        raise TypeError(f"unexpected type {type(_regexp_replace)} for str_replace")



    return _str, _regexp



