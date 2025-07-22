# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
from .._type_aliases import (
    InParamsType,

)

import numbers

from ._agscv_verbose import _val_agscv_verbose
from ._max_shifts import _val_max_shifts
from ._total_passes import _val_total_passes
from ._params import _val_params
from ._total_passes_is_hard import _val_total_passes_is_hard



def _validation(
    _params: InParamsType,
    _total_passes: numbers.Integral,
    _total_passes_is_hard: bool,
    _max_shifts: Union[None, numbers.Integral],
    _agscv_verbose: bool
) -> None:
    """Centralized hub for validation.

    The heavy lifting is handled by the individual submodules. See the
    individual modules for more information.

    Parameters
    ----------
    _params:
        InParamsType
    _total_passes:
        numbers.Integral
    _total_passes_is_hard:
        bool
    _max_shifts:
        Union[None, numbers.Integral]
    _agscv_verbose:
        bool

    Returns
    -------
    None

    """


    _val_total_passes(_total_passes)

    _val_params(_params, _total_passes)

    _val_total_passes_is_hard(_total_passes_is_hard)

    _val_max_shifts(_max_shifts, _can_be_None=True)

    _val_agscv_verbose(_agscv_verbose)








