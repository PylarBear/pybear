# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
from .._type_aliases import (
    DataContainer,
    DoNotDropType,
    ConflictType,
    KeepType,
    FeatureNamesInType
)

import numbers

from ._X import _val_X
from ._conflict import _val_conflict
from ._do_not_drop import _val_do_not_drop
from ._keep import _val_keep
from ._rtol import _val_rtol
from ._atol import _val_atol
from ._equal_nan import _val_equal_nan
from ._n_jobs import _val_n_jobs



def _validation(
    _X: DataContainer,
    _columns: Union[FeatureNamesInType, None],
    _conflict: ConflictType,
    _do_not_drop: DoNotDropType,
    _keep: KeepType,
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
        array-like of shape (n_samples, n_features) - The data.
    _columns:
        Union[FeatureNamesInType, None] - An vector of shape (n_features,)
        if X was passed in a container that has a header, otherwise None.
    _conflict:
        ConflictType
    _do_not_drop:
        DoNotDropType
    _keep:
        KeepType
    _rtol:
        numbers.Real - The relative difference tolerance for equality.
        Must be a non-boolean, non-negative, real number. See
        numpy.allclose.
    _atol:
        numbers.Real - The absolute difference tolerance for equality.
        Must be a non-boolean, non-negative, real number. See
        numpy.allclose.
    _equal_nan:
        bool
    _n_jobs:
        Union[numbers.Integral, None] - The number of joblib Parallel
        jobs to use when scanning the data for columns of constants.


    Return
    ------
    -
        None

    """


    _val_keep(_keep)

    _val_X(_X)

    _val_do_not_drop(_do_not_drop, _X.shape[1], _columns)

    _val_conflict(_conflict)

    _val_rtol(_rtol)

    _val_atol(_atol)

    _val_equal_nan(_equal_nan)

    _val_n_jobs(_n_jobs)






