# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence
from typing_extensions import Union

import numbers

import numpy as np



def _cond_thresholds(
    _thresholds: Union[None, numbers.Real, Sequence[numbers.Real]],  # pizza global ThresholdsType?
) -> list[float]:

    """
    This is a support function for _thresholds__param_grid().

    Condition _thresholds into a 1D list of 1 or more floats.


    Parameters
    ----------
    _thresholds:
        Union[None, numbers.Real, Sequence[numbers.Real]] - user-defined
        threshold(s)


    Return
    ------
    -
        __thresholds: list[float] - user-defined or default floats
        sorted ascending

    """


    try:
        if _thresholds is None:
            __thresholds = \
                list(map(float, np.linspace(0, 1, 21).astype(np.float64)))
            raise MemoryError
        iter(_thresholds)
        # we know from val that its a legit 1D of floats
        __thresholds = list(map(float, set(_thresholds)))
    except MemoryError:
        pass
    except:
        # must be float
        __thresholds = list(map(float, [_thresholds]))


    __thresholds.sort()


    return __thresholds





