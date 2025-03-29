# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Literal, Sequence
from typing_extensions import Union
import numpy.typing as npt
from .._type_aliases import (
    StrType,
    RowSupportType
)

import numpy as np



def _str_1D_core(
    _X: Sequence[str],
    _str_remove: Union[StrType, list[Union[StrType, Literal[False]]]]
) -> tuple[Sequence[str], RowSupportType]:

    """
    Remove unwanted strings from a 1D dataset using exact string matching.


    Parameters
    ----------
     _X:
        Sequence[str] - the data.
    _str_remove:
        Union[StrType, list[Union[StrType, Literal[False]]]] - the removal
        criteria. _str_remove cannot be None. The code that allows entry
        into this module explicitly says "if str_remove is not None:".


    Return
    ------
    -
        tuple[Sequence[str], RowSupportType] - the 1D vector with
        unwanted strings removed and a boolean vector indicating which
        indices were kept (True) and which were removed (False).

    """


    assert isinstance(_X, list)
    assert all(map(isinstance, _X, (str for _ in _X)))


    # _str_remove must be str, set[str], list[Union[str, set[str], False]]
    # _str_remove cannot be None. The code that allows entry into this
    # module explicitly says "if str_remove is not None:".

    if isinstance(_str_remove, str):
        _remove = [_str_remove for _ in range(len(_X))]
    elif isinstance(_str_remove, set):
        _remove = [_str_remove for _ in range(len(_X))]
    elif isinstance(_str_remove, list):
        _remove = _str_remove
    else:
        raise Exception


    _row_support: npt.NDArray[bool] = np.ones(len(_X), dtype=bool)

    for _idx in range(len(_X)-1, -1, -1):

        if _remove[_idx] is False:
            continue

        elif (isinstance(_remove[_idx], str) and _X[_idx] == _remove[_idx]) \
                or (isinstance(_remove[_idx], set) and _X[_idx] in _remove[_idx]):
            _row_support[_idx] = False
            _X.pop(_idx)


    del _remove


    return _X, _row_support







