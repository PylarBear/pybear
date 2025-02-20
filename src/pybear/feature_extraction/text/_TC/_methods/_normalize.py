# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional
from typing_extensions import Union
import numpy.typing as npt

import numpy as np

from ._validation._wip_X import _val_wip_X




def _normalize(
    _WIP_X: Union[list[str], list[list[str]], npt.NDArray[str]],
    _is_2D: bool,
    _upper:Optional[bool] = True       # IF NOT _upper THEN lower
) -> Union[list[str], list[list[str]], npt.NDArray[str]]:

    """
    Set all text in the data to upper case (default) or lower case.


    Parameters
    ----------
    _WIP_X:
        Union[list[str], list[list[str]], npt.NDArray[str]] - the data.
    _is_2D:
        bool - whether the data is 1D or 2D.
    _upper:
        Optional[bool], default=True - the case to normalize to; upper
        case if True, lower case if False.


    Return
    ------
    -
        Union[list[str], list[list[str]], npt.NDArray[str]] - the data
        with normalized text.


    """


    _val_wip_X(_WIP_X)

    if not isinstance(_is_2D, bool):
        raise TypeError(f"'_is_2d' must be boolean")

    if not isinstance(_upper, bool):
        raise TypeError(f"'upper' must be boolean")

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    def _list_normalizer(_X, _upper: bool):

        if _upper:
            return list(map(str.upper, _X))
        else:
            return list(map(str.lower, _X))


    def _np_normalizer(_X, _upper: bool):

        if _upper:
            return np.char.upper(_X)
        else:
            return np.char.lower(_X)


    if _is_2D:
        # WILL PROBABLY BE A RAGGED ARRAY AND np.char WILL THROW A FIT
        # SO GO ROW BY ROW
        if isinstance(_WIP_X, list):
            for row_idx in range(len(_WIP_X)):
                _WIP_X[row_idx] = list(_list_normalizer(_WIP_X[row_idx], _upper))
        elif isinstance(_WIP_X, np.ndarray):
            for row_idx in range(len(_WIP_X)):
                _WIP_X[row_idx] = list(_np_normalizer(_WIP_X[row_idx], _upper))

            # pizza doesnt like the for loop,
            # and hates that it is a ndarray of lists! but if leave as
            # ndarray it raises for setting array element with a sequence.
            # see if u can get it to just
            # _WIP_X = _nd_array_stripper(_WIP_X)

    elif not _is_2D:   # LIST OF strs
        if isinstance(_WIP_X, list):
            _WIP_X = _list_normalizer(_WIP_X, _upper)
        elif isinstance(_WIP_X, np.ndarray):
            _WIP_X = _np_normalizer(_WIP_X, _upper)
        else:
            raise Exception


    del _list_normalizer, _np_normalizer


    return _WIP_X








