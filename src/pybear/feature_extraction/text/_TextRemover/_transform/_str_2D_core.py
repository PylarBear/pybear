# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import (
    XContainer,
    StrRemoveType
)



def _str_2D_core(
    _X: XContainer,
    _str_remove: StrRemoveType
) -> XContainer:

    """
    Remove unwanted strings from a 2D dataset using exact string matching.


    Parameters
    ----------
     _X:
        XContainer - the data.
    _str_remove:
        StrRemoveType - the removal criteria.


    Return
    ------
    -
        XContainer

    """

    assert isinstance(_X, list)
    assert isinstance(_X[0], list)
    assert isinstance(_X[0][0], str)


    if isinstance(_str_remove, str):
        _remove = [_str_remove for _ in _X]
    elif isinstance(_str_remove, set):
        _remove = [_str_remove for _ in _X]
    elif isinstance(_str_remove, list):
        _remove = _str_remove
    else:
        raise Exception


    for _idx in range(len(_X)-1, -1, -1):

        if _remove[_idx] is False:
            continue

        elif isinstance(_remove[_idx], str):
            while _remove[_idx] in _X[_idx]:
                _X[_idx].remove(_remove[_idx])

        elif isinstance(_remove[_idx], set):
            for __ in _remove[_idx]:
                while __ in _X[_idx]:
                    _X[_idx].remove(__)
        else:
            raise Exception

        if len(_X[_idx]) == 0:
            _X.pop(_idx)


    del _remove


    return _X










