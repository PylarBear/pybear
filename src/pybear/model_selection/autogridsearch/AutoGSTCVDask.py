# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence
from typing_extensions import Union

import numbers

from .autogridsearch_wrapper import autogridsearch_wrapper
from . import autogridsearch_docs

from ..GSTCV._GSTCVDask.GSTCVDask import GSTCVDask



class AutoGSTCVDask(autogridsearch_wrapper(GSTCVDask)):

    #     AutoGridSearchCV = type(
    #         'SklearnAutoGridSearch',
    #         (autogridsearch_wrapper(GSTCVDask),),
    #         {'__doc__': autogridsearch_docs,
    #          '__init__.__doc__': autogridsearch_docs}
    #     )


    __doc__ = autogridsearch_docs.__doc__


    def __init__(
        self,
        estimator,
        params: dict[
            str,
            list[Union[Sequence[any]], Union[int, Sequence[int]], str]
        ],
        *,
        total_passes: int = 5,
        total_passes_is_hard: bool = False,
        max_shifts: Union[None, numbers.Integral] = None,
        agscv_verbose: bool = False,
        **parent_gscv_kwargs
    ):

        __doc__ = autogridsearch_docs.__doc__

        super().__init__(
            estimator,
            params,
            total_passes=total_passes,
            total_passes_is_hard=total_passes_is_hard,
            max_shifts=max_shifts,
            agscv_verbose=agscv_verbose,
            **parent_gscv_kwargs
        )








