# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from ._keep_and_columns import _val_keep_and_columns
from ._equal_nan import _val_equal_nan
from ._rtol import _val_rtol
from ._atol import _val_atol
from ._n_jobs import _val_n_jobs
from ._X import _val_X



from .._type_aliases import (
    ColumnsType,  # pizza
    DataType,
    KeepType
)
from typing_extensions import Union


def _validation(
    _X:DataType,
    _columns: ColumnsType,   # pizza think if we want to just make them all explicit
    _keep: KeepType,
    _equal_nan: bool,
    _rtol: float,
    _atol: float,
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
        pizza WHAT!?!?!?!!
    _keep:
        Literal['first', 'last', 'random'], dict[str, int], None
    _equal_nan:
        bool,
    _rtol:
        float,
    _atol:
        float,
    _n_jobs:
        Union[int, None]


    Return
    ------
    -
        None


    """


    _val_X(_X)

    _val_keep_and_columns(_keep, _columns, _X)

    _val_equal_nan(_equal_nan)

    _val_rtol(_rtol)

    _val_atol(_atol)

    _val_n_jobs(_n_jobs)






