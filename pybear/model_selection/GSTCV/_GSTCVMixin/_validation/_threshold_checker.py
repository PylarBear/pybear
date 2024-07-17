# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing import Union, Iterable
import numpy as np
import numpy.typing as npt



def _threshold_checker(
        __thresholds: Union[None, Iterable[Union[int, float]], Union[int, float]],
        is_from_kwargs: bool,
        idx: int
    ) -> npt.NDArray[np.float64]:

    if not isinstance(is_from_kwargs, bool):
        raise TypeError(f"'is_from_kwargs' must be a bool")

    try:
        float(idx)
        if isinstance(idx, bool):
            raise Exception
        if not int(idx) == idx:
            raise Exception
        idx = int(idx)
        if idx < 0:
            raise Exception
    except:
        raise TypeError(f"'idx' must be an int >= 0")


    # error messaging *** *** *** *** *** *** *** *** *** *** *** ***
    base_msg = (f"must be (1 - a list-type of 1 or more numbers or "
                f"2 - a single number) and 0 <= number(s) <= 1")
    if is_from_kwargs:
        err_msg = f"thresholds passed as a kwarg " + base_msg
    else:
        err_msg = f"thresholds passed as param to param_grid[{idx}] " + base_msg
    del base_msg
    # END error messaging *** *** *** *** *** *** *** *** *** *** ***

    # outer container * * * * * * * * * * * * * * * * * * * * * * * *
    if __thresholds is None:
        __thresholds = np.linspace(0, 1, 21).astype(np.float64)

    try:
        iter(__thresholds)
        if isinstance(__thresholds, (str, dict)):
            raise Exception
        __thresholds = list(set(__thresholds))
    except:
        try:
            float(__thresholds)
            if isinstance(__thresholds, bool):
                raise Exception
            __thresholds = [__thresholds]
        except:
            raise TypeError(err_msg)

    try:
        __thresholds = np.array(list(__thresholds), dtype=np.float64)
    except:
        raise TypeError(err_msg)

    if len(__thresholds) == 0:
        raise ValueError(err_msg)
    # outer container * * * * * * * * * * * * * * * * * * * * * * * *

    # inner objects ... --- ... ... --- ... ... --- ... ... --- ...
    for _thresh in __thresholds:
        try:
            float(_thresh)
        except:
            raise TypeError(err_msg)

        if not (_thresh >= 0 and _thresh <= 1):
            raise ValueError(err_msg)
    # END inner objects ... --- ... ... --- ... ... --- ... ... --- ...

    __thresholds.sort()

    del err_msg
    return __thresholds






























