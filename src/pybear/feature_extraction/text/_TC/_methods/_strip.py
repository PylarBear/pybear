# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
import numpy.typing as npt

import re

import numpy as np

from ._validation._wip_X import _val_wip_X



def _strip(
    _WIP_X: Union[list[str], list[list[str]], npt.NDArray[str]],
    _is_2D: bool
) -> Union[list[str], list[list[str]], npt.NDArray[str]]:

    """
    Remove multiple spaces and leading and trailing spaces from all text
    in the data.


    Parameters
    ----------
    _WIP_X:
        Union[list[str], list[list[str]], npt.NDArray[str]] - The data
        object. Must be a list of strings, a list of lists of strings,
        or a numpy array of strings.
    _is_2D:
        bool - whether the data object is 1D or 2D.


    Return
    ------
    -
        Union[list[str], list[list[str]], npt.NDArray[str]] - the data
        less any unnecessary spaces.

    """


    _val_wip_X(_WIP_X)

    if not isinstance(_is_2D, bool):
        raise TypeError(f"'_is_2D' must be boolean")

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    def _list_stripper(_X):
        for _idx, _string in enumerate(_X):
            _string = re.sub(" +", " ", _string)
            _string = re.sub(" ,", ",", _string)
            _X[_idx] = _string.strip()
        return list(_X)


    def _ndarray_stripper(_X):
        # pizza put .astype(str) to get this to pass on object dtype
        # is there a more robust way
        while not np.all(np.char.find(_X.astype(str), "  ") == -1):
            _X = np.char.replace(_X.astype(str), f'  ', f' ')

        _X = np.char.replace(_X.astype(str), f' ,', f',')

        _X = np.char.strip(_X)
        return np.array(_X)


    if _is_2D:

        # DO THIS ROW-WISE (SINGLE ARRAY AT A TIME), BECAUSE np.char WILL
        # THROW A FIT IF GIVEN A RAGGED ARRAY

        if isinstance(_WIP_X, list):

            for row_idx in range(len(_WIP_X)):
                _WIP_X[row_idx] = _list_stripper(_WIP_X[row_idx])

        elif isinstance(_WIP_X, np.ndarray):

            for row_idx in range(len(_WIP_X)):
                _WIP_X[row_idx] = _ndarray_stripper(_WIP_X[row_idx])

            # pizza doesnt like the for loop, see if u can get it to just
            # _WIP_X = _ndarray_stripper(_WIP_X)


    elif not _is_2D:  # MUST BE 1D OF strs

        if isinstance(_WIP_X, list):

            _WIP_X = _list_stripper(_WIP_X)

        elif isinstance(_WIP_X, np.ndarray):

            _WIP_X = _ndarray_stripper(_WIP_X)

        else:
            raise Exception


    del _list_stripper, _ndarray_stripper


    return _WIP_X






