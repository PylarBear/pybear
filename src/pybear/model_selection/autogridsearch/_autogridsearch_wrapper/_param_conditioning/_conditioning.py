# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import (
    InParamsType,
    ParamsType
)

import numbers

from ._total_passes import _cond_total_passes
from ._params import _cond_params
from ._max_shifts import _cond_max_shifts



def _conditioning(
    _params: InParamsType,
    _total_passes: numbers.Integral,
    _max_shifts: numbers.Integral,
    _inf_max_shifts: numbers.Integral
) -> tuple[ParamsType, int, int]:

    """
    Centralized hub for conditioning parameters. Condition given
    `max_shifts`, `params`, and `total_passes` into internal processing
    containers, types, and values.


    Parameters
    ----------
    _params:
        InParamsType
    _total_passes:
        numbers.Integral
    _max_shifts:
        numbers.Integral
    _inf_max_shifts:
        numbers.Integral


    Returns
    -------
    -
        _params: ParamsType - the conditioned params. all sequences
        converted to python list, any Nones in the 'points' slots
        for string or bool params converted to a large integer. any
        integers in the points slots for numeric params converted to
        lists.

        _total_passes: numbers.Integral - the conditioned total_passes,
        a python integer.

        _max_shifts: numbers.Integral - the conditioned max_shifts; set
        to a large integer if passed as None.


    """


    _total_passes = _cond_total_passes(_total_passes)

    _params = _cond_params(_params, _total_passes)

    _max_shifts = _cond_max_shifts(_max_shifts, _inf_max_shifts)


    return _params, _total_passes, _max_shifts



