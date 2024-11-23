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
    DataFormatType,
    KeepType
)
from typing import Iterable
from typing_extensions import Union
from numbers import Real


def _validation(
    _X:DataFormatType,
    _columns: Union[Iterable[str], None],
    _keep: KeepType,
    _equal_nan: bool,
    _rtol: Real,
    _atol: Real,
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
        Literal['first', 'last', 'random', 'none'], dict[str, int], str, int
    _equal_nan:
        bool,
    _rtol:
        Real,
    _atol:
        Real,
    _n_jobs:
        Union[int, None]


    Return
    ------
    -
        None


    """


    _val_keep_and_columns(_keep, _columns, _X)

    _val_equal_nan(_equal_nan)

    _val_rtol(_rtol)

    _val_atol(_atol)

    _val_n_jobs(_n_jobs)






