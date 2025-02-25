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



def _str_1D_core(
    _X: XContainer,
    _str_remove: StrRemoveType
) -> tuple[XContainer, RowSupportType]:

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
        tuple[XContainer, RowSupportType] - the 1D vector with unwanted
        strings removed and a boolean vector indicating which rows were
        kept.

    """


    assert isinstance(_X, list)
    assert all(map(isinstance, _X, (str for _ in _X)))


    # _str_remove must be str, set[str], list[Union[str, set[str], False]]

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
            if _X[_idx] == _remove[_idx]:
                _row_support[_idx] = False
                _X.pop(_idx)

        elif isinstance(_remove[_idx], set):
            for __ in _remove[_idx]:
                if _X[_idx] == __:
                    _row_support[_idx] = False
                    _X.pop(_idx)
                    break
        else:
            raise Exception


    del _remove


    return _X, _row_support







