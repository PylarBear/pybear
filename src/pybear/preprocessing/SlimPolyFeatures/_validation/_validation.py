# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Callable


from ._atol import _val_atol
from ._degree__min_degree import _val_degree__min_degree
from ._equal_nan import _val_equal_nan
from ._feature_name_combiner import _val_feature_name_combiner
from ._interaction_only import _val_interaction_only
from ._keep import _val_keep
from ._n_jobs import _val_n_jobs
from ._rtol import _val_rtol
from ._scan_X import _val_scan_X
from ._sparse_output import _val_sparse_output
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
    _feature_name_combiner: Callable[[Iterable[str], tuple[int, ...]], str],
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
    _feature_name_combiner:
        Callable[[tuple[int, ...]], str] -
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

    # dont need _val_X here, _val_X is before _validation() in both partial_fit() & transform()

    _val_scan_X(_scan_X)

    _val_equal_nan(_equal_nan)

    _val_n_jobs(_n_jobs)

    _val_rtol(_rtol)

    _val_degree__min_degree(_degree, _min_degree)

    _val_feature_name_combiner(
        _feature_name_combiner,
        _min_degree,
        _degree,
        _X.shape[1],
        _interaction_only
    )

    _val_interaction_only(_interaction_only)

    _val_sparse_output(_sparse_output)
















