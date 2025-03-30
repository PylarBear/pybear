# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import (
    XContainer,
    RegExpReplaceType,
    RegExpReplaceArgsType
)

from copy import deepcopy



def _regexp_param_conditioner(
    _regexp_replace: RegExpReplaceType,
    _X: XContainer
) -> list[set[RegExpReplaceArgsType]]:

    """
    Standardize the 'regexp_replace' parameters into lists of sets of
    args for re.sub.


    Parameters
    ----------
    _regexp_replace:
        RegExpReplaceType - the re.sub parameters.


    Return
    ------
    -
        list[set[RegExpReplaceArgsType]] - a list of sets of the re.sub
        parameters for each rowin X.


    """


    if _regexp_replace is None:
        _regexp = [set() for _ in _X]
    elif _regexp_replace is False:
        # this is a fail-safe. rr could only be False if X is 2D, rr is
        # a list, and we have sent one of the rows of X and its rr value
        # into here. but in main TR.transform False is explicitly skipped
        # pizza u cant pass a bool to str.replace! this needs to be blocked!
        _regexp = [set((_regexp_replace,)) for _ in _X]
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
        raise TypeError(f"unexpected type {type(_regexp_replace)} for regexp_replace")



    return _regexp



