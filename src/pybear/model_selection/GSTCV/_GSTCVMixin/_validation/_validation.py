# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Iterable, Literal
from typing_extensions import Union
from ..._type_aliases import (
    GenericKFoldType,
    ScorerInputType
)

import numbers

from ._scoring import _val_scoring
from ._n_jobs import _val_n_jobs
from ._cv import _val_cv
from ._verbose import _val_verbose
from ._error_score import _val_error_score
from ._return_train_score import _val_return_train_score



def _validation(
    _scoring: ScorerInputType,
    _n_jobs: Union[numbers.Integral, None],
    _cv: Union[None, numbers.Integral, Iterable[GenericKFoldType]],
    _verbose: numbers.Real,
    _error_score: Union[Literal['raise'], numbers.Real],
    _return_train_score: bool
) -> None:

    """
    Centralized hub for validation. See the individual submodules for
    more information.
    
    
    Parameters
    ----------
    _scoring:
        ScorerInputType
    _n_jobs:
        Union[numbers.Integral, None]
    _cv:
        Union[None, numbers.Integral, Iterable[GenericKFoldType]]
    _verbose:
        numbers.Real
    _error_score:
        Union[numbers.Real, Literal['raise']]
    _return_train_score:
        bool


    Returns
    -------
    -
        None

    """


    _val_scoring(_scoring)

    _val_n_jobs(_n_jobs)

    _val_cv(_cv, _can_be_None=True, _can_be_int=True)

    _val_verbose(_verbose, _can_be_raw_value=True)

    _val_error_score(_error_score)

    _val_return_train_score(_return_train_score)






