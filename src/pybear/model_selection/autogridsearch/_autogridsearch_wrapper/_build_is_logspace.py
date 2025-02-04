# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Literal
from typing_extensions import Union

import numbers
import numpy as np

from ._type_aliases import ParamsType



def _build_is_logspace(
        _params: ParamsType
    ) -> dict[str, Union[Literal[False], numbers.Real]]:

    """
    _IS_LOGSPACE is a dictionary keyed by all param names, including
    string params. String params are always False. For numerical params,
    if the space is linear, or some other non-standard interval, it is
    False. If it is logspace, the 'truth' of being a logspace is
    represented by a number indicating the interval of the logspace.
    E.g., np.logspace(-5, 5, 11) would be represented by 1.0, and
    np.logspace(-20, 20, 9) would be represented by 5.0.


    Parameters
    ----------
    _params:
        dict[str, list[GridType, PointsType, str]] - autogridsearch's
        instructions for performing grid searches for each parameter.


    Return
    ------
    -
        _IS_LOGSPACE: dict[str, Union[Literal[False], numbers.Real]] -
        a dictionary indicating whether a parameters search space is
        logarithmic. If so, the logspace interval of the space.


    """

    _IS_LOGSPACE = dict()
    for _param in _params:

        __ = _params[_param]

        if __[-1] == 'string':
            _IS_LOGSPACE[_param] = False

        elif __[-1] == 'bool':
            _IS_LOGSPACE[_param] = False

        else:
            # "soft" & "hard" CAN BE LOGSPACES, BUT "fixed" CANNOT
            if "fixed" in __[-1]:
                _IS_LOGSPACE[_param] = False
                continue

            # if 0 in the space, cannot be logspace
            if 0 in __[0]:
                _IS_LOGSPACE[_param] = False
                continue

            # if 2 or less points in points, cannot be logspace
            if __[-2][0] <= 2:
                _IS_LOGSPACE[_param] = False
                continue

            # IF IS LOGSPACE, PUT IN THE SIZE OF THE GAP (bool(>0) WILL RETURN True)
            log_gap = np.log10(__[0])[1:] - np.log10(__[0])[:-1]

            if len(np.unique(log_gap)) == 1:  # UNIFORM GAP SIZE IN LOG SCALE
                _IS_LOGSPACE[_param] = log_gap[0]
            else:
                _IS_LOGSPACE[_param] = False  # MUST BE LINSPACE OR SOMETHING ELSE

            del __, log_gap



    return _IS_LOGSPACE








