# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union

import numbers



def _cond_max_shifts(
    _max_shifts: Union[None, numbers.Integral]
) -> numbers.Integral:

    """
    When 'max_shifts' is passed as None to agscv, this indicates
    unlimited shifts allowed. Condition 'max_shifts' into a large number.


    Parameters
    ----------
    _max_shifts:
        Union[None, numbers.Integral] - the maximum number of grid-shift
        passes agscv is allowed to make. If None, the number of shifting
        passes allowed is unlimited.


    Returns
    -------
    -
        numbers.Integral: a large number that should never be attainable.


    """

    # cannot use float('inf') here, validation wants numbers.Integral
    return _max_shifts or 1_000



