# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



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
from ._X import _val_X   # pizza

from .._type_aliases import DataType, FeatureNameCombinerType
from typing import Literal
from typing_extensions import Union

import numbers




def _validation(
    _X:DataType,
    _degree: numbers.Integral,
    _min_degree: numbers.Integral,
    _scan_X: bool,
    _keep: Literal['first', 'last', 'random'],
    _interaction_only: bool,
    _sparse_output: bool,
    _feature_name_combiner: FeatureNameCombinerType,
    _rtol: numbers.Real,
    _atol: numbers.Real,
    _equal_nan: bool,
    _n_jobs: Union[numbers.Integral, None]
) -> None:

    """
    Centralized hub for performing parameter validation.
    See the individual modules for more information.


    Parameters
    ----------
    _X:
        {array-like, scipy sparse matrix} of shape (n_samples, n_features)
    _degree:
        numbers.Integral
    _min_degree:
        numbers.Integral
    _scan_X:
        bool
    _keep:
        Literal['first', 'last', 'random']
    _interaction_only:
        bool
    _sparse_output:
        bool
    _feature_name_combiner:
        Union[
            Callable[[Iterable[str], tuple[int, ...]], str],
            Literal['as_feature_names', 'as_indices']]
        ]
    _rtol:
        numbers.Real
    _atol:
        numbers.Real
    _equal_nan:
        bool
    _n_jobs:
        Union[numbers.Integral, None]


    Return
    ------
    -
        None


    """

    _val_keep(_keep)

    # pizza, _val_X will probably end up back in here, because of sk _validate_data coming out.

    _val_scan_X(_scan_X)

    _val_degree__min_degree(_degree, _min_degree)

    _val_feature_name_combiner(_feature_name_combiner)

    _val_interaction_only(_interaction_only)

    _val_sparse_output(_sparse_output)

    _val_equal_nan(_equal_nan)

    _val_rtol(_rtol)

    _val_atol(_atol)

    _val_n_jobs(_n_jobs)














