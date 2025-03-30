# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import TRStrReplaceArgsType

from ._str_param_conditioner import _str_param_conditioner



def _str_1D_core(
    _X: list[str],
    _str_replace: TRStrReplaceArgsType
) -> list[str]:

    """
    Search and replace strings in a 1D dataset using exact string
    matching.


    Parameters
    ----------
     _X:
        list[str] - the data or a row of data.
    _str_replace:
        Union[tuple[str, str], tuple[str, str, int]] - the search and
        replace criteria. must have been conditioned by _param_conditioner
        before being passed here.


    Return
    ------
    -
        list[str] - the 1D vector with string substitutions made.


    """

    print(f'pizza test {_X=}')
    assert isinstance(_X, list)
    assert all(map(isinstance, _X, (str for _ in _X)))

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    __str_replace = _str_param_conditioner(_str_replace, _X)

    for _idx in range(len(_X)):

        # if set is empty will skip (False in og list becomes empty set)
        for _tuple in __str_replace[_idx]:   # set of tuples (or empty)

            # for str_replace, the tuples are already the exact args for str.replace
            _X[_idx] = _X[_idx].replace(*_tuple)


    return _X







