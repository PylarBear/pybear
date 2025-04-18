# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union

import numbers



def _val_max_shifts(
    _max_shifts: Union[None, numbers.Integral]
) -> None:


    """
    Validate _max_shifts. Can be None or an integer >= 1.


    Parameters
    ----------
    _max_shifts:
        Union[None, numbers.Integral] - the maximum number of grid shifts
        allowed when trying to center parameters within their search
        grids.


    Returns
    -------
    -
        None

    """


    if _max_shifts is None:
        return


    err_msg = f"if passed, 'max_shifts'  must be an integer >= 1. \ngot {_max_shifts}."

    if not isinstance(_max_shifts, numbers.Integral) or isinstance(_max_shifts, bool):
        raise TypeError(err_msg)

    if _max_shifts < 1:
        raise ValueError(err_msg)





