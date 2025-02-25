# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import (
    XContainer,
    StrRemoveType
)


def _str_1D_core(
    _X: XContainer,
    _str_remove: StrRemoveType
) -> XContainer:

    """
    Remove unwanted strings from a 1D dataset using exact string matching.


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
    assert isinstance(_X[0], str)


    # _str_remove must be str, set[str], list[Union[str, set[str], False]]

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
            if _X[_idx] == _remove[_idx]:
                _X.pop(_idx)

        elif isinstance(_remove[_idx], set):
            for __ in _remove[_idx]:
                if _X[_idx] == __:
                    _X.pop(_idx)
                    break
        else:
            raise Exception


    del _remove


    return _X







