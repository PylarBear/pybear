# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional
from typing_extensions import Union

import numbers



def _val_max_shifts(
    _max_shifts: Union[None, numbers.Integral],
    _can_be_None: Optional[bool] = False
) -> None:
    """Validate `_max_shifts`.

    Must be an integer >= 1. Can conditionally be None.

    Parameters
    ----------
    _max_shifts : Union[None, numbers.Integral]
        The maximum number of grid shifts allowed when trying to center
        parameters within their search grids.
    _can_be_None : Optional[bool], default=False
        Whether `max_shifts` can be None-valued.

    Returns
    -------
    None

    """


    if _can_be_None and _max_shifts is None:
        return

    if _can_be_None:
        err_msg = \
            f"'max_shifts'  must be None or an integer >= 1. \ngot {_max_shifts}."
    elif not _can_be_None:
        err_msg = f"'max_shifts'  must be an integer >= 1. \ngot {_max_shifts}."

    if not isinstance(_max_shifts, numbers.Integral) \
            or isinstance(_max_shifts, bool):
        raise TypeError(err_msg)

    if _max_shifts < 1:
        raise ValueError(err_msg)





