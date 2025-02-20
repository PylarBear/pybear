# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence
from typing_extensions import Union
from .._type_aliases import WIPXContainer

import numpy as np



def _strip(
    _WIP_X: WIPXContainer,
    _is_2D: bool
) -> WIPXContainer:

    """
    Remove extra spaces from the data.


    Parameters
    ----------
    _WIP_X:
        Union[Sequence[str], Sequence[Sequence[str]]] - The data object.
    _is_2D:
        bool - whether the data object is 1D or 2D.

    Return
    ------
    -
        Union[Sequence[str], Sequence[Sequence[str]]]

    """


    # DO THIS ROW-WISE (SINGLE ARRAY AT A TIME), BECAUSE np.char WILL
    # THROW A FIT IF GIVEN A RAGGED ARRAY

    # pizza u will be back. 2D needs work. think about container agnostic.

    # pizza _WIP_X needs validation
    # _WIP_X: WIPXContainer
    if not isinstance(_is_2D, bool):
        raise TypeError(f"'_is_2D' must be boolean")



    if _is_2D:
        for row_idx in range(len(_WIP_X)):
            for idx in range(len(_WIP_X[row_idx])):
                while f'  ' in _WIP_X[row_idx][idx]:
                    _WIP_X[row_idx][idx] = \
                        np.char.replace(_WIP_X[row_idx][idx], f'  ', f' ')

                _WIP_X[row_idx][idx] = np.char.strip(_WIP_X[row_idx][idx])

    elif not _is_2D:  # MUST BE LIST OF strs
        for idx in range(len(_WIP_X)):
            while f'  ' in _WIP_X[idx]:
                _WIP_X[idx] = str(np.char.replace(_WIP_X[idx], f'  ', f' '))

            _WIP_X[idx] = np.char.strip(_WIP_X[idx])


    return _WIP_X






