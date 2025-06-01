# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Literal
from typing_extensions import Union
from .._type_aliases import FeatureNameCombinerType
from ...__shared._type_aliases import XContainer

import numbers
import warnings

from ._degree__min_degree import _val_degree__min_degree
from ._feature_name_combiner import _val_feature_name_combiner
from ._keep import _val_keep
from ._X_supplemental import _val_X_supplemental

from ...__shared._validation._X import _val_X
from ...__shared._validation._equal_nan import _val_equal_nan
from ...__shared._validation._atol import _val_atol
from ...__shared._validation._rtol import _val_rtol
from ...__shared._validation._n_jobs import _val_n_jobs
from ...__shared._validation._any_bool import _val_any_bool
from ...__shared._validation._any_integer import _val_any_integer



def _validation(
    _X: XContainer,
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
    _n_jobs: Union[numbers.Integral, None],
    _job_size: numbers.Integral
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
            Callable[[Sequence[str], tuple[int, ...]], str],
            Literal['as_feature_names', 'as_indices']]
        ]
    _rtol:
        numbers.Real
    _atol:
        numbers.Real
    _equal_nan:
        bool
    _n_jobs:
        Union[numbers.Integral, None] - The number of joblib Parallel
        jobs to use when scanning the data for duplicate columns.
    _job_size:
        numbers.Integral - The number of columns to send to a joblib job.
        Must be an integer greater than or equal to 2.


    Return
    ------
    -
        None

    """


    _val_keep(_keep)

    _val_any_bool(_scan_X, 'scan_X', _can_be_None=False)

    if _scan_X is False:
        warnings.warn(
            f"'scan_X' is set to False. Do this with caution, only when "
            f"you are certain that X does not have constant or duplicate "
            f"columns. Otherwise the results from :meth: 'transform' will "
            f"be nonsensical."
        )

    _val_degree__min_degree(_degree, _min_degree)

    _val_feature_name_combiner(_feature_name_combiner)

    _val_any_bool(_interaction_only, 'interaction_only', _can_be_None=False)

    _val_any_bool(_sparse_output, 'sparse_output', _can_be_None=False)

    _val_equal_nan(_equal_nan)

    _val_rtol(_rtol)

    _val_atol(_atol)

    _val_n_jobs(_n_jobs)

    # _val_any_integer allows lists
    if not isinstance(_job_size, numbers.Integral):
        raise TypeError(f"'job_size' must be an integer >= 2. Got {_job_size}.")
    _val_any_integer(_job_size, 'job_size', _min=2)

    _val_X(_X)

    _val_X_supplemental(_X, _interaction_only)



