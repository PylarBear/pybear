# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy.typing as npt
from .._type_aliases import (
    XContainer,
    StrRemoveType,
    RowSupportType
)

import numpy as np



def _str_2D_core(
    _X: XContainer,
    _str_remove: StrRemoveType
) -> tuple[XContainer, RowSupportType]:

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
        tuple[XContainer, RowSupportType] - the 2D array of data with
        unwanted strings removed and the boolean vector indicating which
        rows in the data were kept.

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


    _row_support: npt.NDArray[bool] = np.ones(len(_X), dtype=bool)

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
            _row_support[_idx] = False
            _X.pop(_idx)


    del _remove


    return _X, _row_support










