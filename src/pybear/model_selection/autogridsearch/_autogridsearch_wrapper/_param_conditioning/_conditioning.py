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

from ._params__total_passes import _cond_params__total_passes
from ._max_shifts import _cond_max_shifts



def _conditioning(
    _params: InParamsType,
    _total_passes: numbers.Integral,
    _max_shifts: numbers.Integral,
    _inf_shrink_pass: numbers.Integral,
    _inf_max_shifts: numbers.Integral
) -> tuple[ParamsType, numbers.Integral, numbers.Integral]:

    """
    Centralized hub for conditioning parameters. Condition given
    `max_shifts`, `params`, and `total_passes` into internal processing
    containers and values.


    Parameters
    ----------
    _params
    _total_passes
    _max_shifts
    _inf_shrink_pass
    _inf_max_shifts


    Returns
    -------
    -
        _params: ParamsType - the conditioned params. all sequences
        converted to python list, any Nones in the 'points' slots
        for string or bool params converted to a large integer. any
        integers in the points slots for numeric params converted to
        lists.

        _total_passes: numbers.Integral - the conditioned total_passes.
        Pizza as of 25_04_21 this number is changed to match the length
        of any list-like 'points' passed to numerical params. think on
        if we want this.

        _max_shifts: numbers.Integral - the conditioned max_shifts; set
        to a large integer if passed as None. pizza maybe we just leave
        the Nones. think on it.


    """


    _params, _total_passes = _cond_params__total_passes(
        _params,
        _total_passes,
        _inf_shrink_pass
    )

    _max_shifts = _cond_max_shifts(_max_shifts, _inf_max_shifts)


    return _params, _total_passes, _max_shifts



