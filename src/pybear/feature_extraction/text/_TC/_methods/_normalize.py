# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional
from .._type_aliases import WIPXContainer

import numpy as np



def _normalize(
    _WIP_X: WIPXContainer,
    _is_list_of_lists: bool,
    _upper:Optional[bool] = True
) -> WIPXContainer:    # IF NOT _upper THEN lower

    """
    Set all text in CLEANED_TEXT object to upper case (default) or
    lower case.


    Parameters
    ----------
    _WIP_X:
        WIPXContainer - the data.
    _is_2D:
        bool - whether the data is 1D or 2D.
    _upper:
        Optional[bool], default=True - the case to normalize to; _upper
        case if True, lower-case if False.


    Return
    ------
    -
        pizza?


    """
    # WILL PROBABLY BE A RAGGED ARRAY AND np.char WILL THROW A FIT, SO GO ROW BY ROW

    if not isinstance(_upper, bool):
        raise TypeError(f"'upper' must be boolean")

    if _is_list_of_lists:
        for row_idx in range(len(_WIP_X)):
            if _upper:
                _WIP_X[row_idx] = \
                    np.fromiter(
                        map(str.upper, _WIP_X[row_idx]),
                        dtype='U30'
                    )
            elif not _upper:
                _WIP_X[row_idx] = \
                    np.fromiter(
                        map(str.lower, _WIP_X[row_idx]),
                        dtype='U30'
                    )
    elif not _is_list_of_lists:   # LIST OF strs
        if _upper:
            _WIP_X = \
                np.fromiter(
                    map(str.upper, _WIP_X),
                    dtype='U100000'
                )
        elif not _upper:
            _WIP_X = \
                np.fromiter(
                    map(str.lower, _WIP_X),
                    dtype='U100000'
                )


    return _WIP_X
