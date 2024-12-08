# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from ._atol import _val_atol
from ._degree__min_degree import _val_degree__min_degree
from ._equal_nan import _val_equal_nan
from ._scan_X import _val_scan_X
from ._interaction_only import _val_interaction_only
from ._keep import _val_keep
from ._n_jobs import _val_n_jobs
from ._sparse_output import _val_sparse_output
from ._rtol import _val_rtol
from ._X import _val_X

from .._type_aliases import DataType
from typing import Literal, Iterable
from typing_extensions import Union

import numbers


def _validation(
    _X:DataType,
    _columns: Union[Iterable[str], None],
    _degree: int,
    _min_degree: int,
    _scan_X: bool,
    _keep: Literal['first', 'last', 'random'],
    _interaction_only: bool,
    _sparse_output: bool,
    _rtol: numbers.Real,
    _atol: numbers.Real,
    _equal_nan: bool,
    _n_jobs: Union[int, None]
) -> None:

    """
    Centralized hub for performing parameter validation.
    See the individual modules for more information.


    Parameters
    ----------
    _X:
        {array-like, scipy sparse matrix} of shape (n_samples, n_features)
    _columns:
        Union[Iterable[str], None]
    _degree:
        int
    _min_degree:
        int
    _scan_X:
        bool
    _keep:
        Literal['first', 'last', 'random']
    _interaction_only:
        bool
    _sparse_output:
        bool
    _rtol:
        float
    _atol:
        float
    _equal_nan:
        bool
    _n_jobs:
        Union[int, None]


    Return
    ------
    -
        None


    """


    _val_atol(_atol)

    _val_keep(_keep)

    _val_X(_X)

    _val_scan_X(_scan_X)

    _val_equal_nan(_equal_nan)

    _val_n_jobs(_n_jobs)

    _val_rtol(_rtol)

    _val_degree__min_degree(_degree, _min_degree)

    _val_interaction_only(_interaction_only)

    _val_sparse_output(_sparse_output)
















