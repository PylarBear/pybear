# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import (
    Any,
    Sequence
)

import numbers

from .autogridsearch_wrapper import autogridsearch_wrapper
from . import autogridsearch_docs

from ..GSTCV._GSTCV.GSTCV import GSTCV



class AutoGSTCV(autogridsearch_wrapper(GSTCV)):


    __doc__ = autogridsearch_docs.__doc__


    def __init__(
        self,
        estimator,
        params: dict[
            str,
            Sequence[tuple[
                Sequence[Any],
                numbers.Integral | Sequence[numbers.Integral],
                str
            ]]
        ],
        *,
        total_passes:numbers.Integral = 5,
        total_passes_is_hard:bool = False,
        max_shifts:numbers.Integral | None = None,
        agscv_verbose:bool = False,
        **parent_gscv_kwargs
    ):
        """Initialize the `AutoGSTCV` instance."""

        super().__init__(
            estimator,
            params,
            total_passes=total_passes,
            total_passes_is_hard=total_passes_is_hard,
            max_shifts=max_shifts,
            agscv_verbose=agscv_verbose,
            **parent_gscv_kwargs
        )







