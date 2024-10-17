# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from ._conflict import _val_conflict
from ._do_not_drop import _val_do_not_drop
from ._keep import _val_keep
from ._rtol import _val_rtol
from ._atol import _val_atol
from ._equal_nan import _val_equal_nan
from ._n_jobs import _val_n_jobs
from ._X import _val_X



from .._type_aliases import (
    ColumnsType,
    ConflictType,
    DataType,
    DoNotDropType,
    KeepType
)
from typing_extensions import Union


def _validation(
    _X:DataType,
    _columns: ColumnsType,
    _conflict: ConflictType,
    _do_not_drop: DoNotDropType,
    _keep: KeepType,
    _rtol: float,
    _atol: float,
    _equal_nan: bool,
    _n_jobs: Union[int, None]
) -> None:

    """
    Pizza finish!


    Parameters
    ----------
    _X:DataType,
    _columns: ColumnsType,
    _conflict: ConflictType,
    _do_not_drop: DoNotDropType,
    _keep: KeepType,
    _rtol: float,
    _atol: float,
    _equal_nan: bool,
    _n_jobs: Union[int, None]

    Return
    ------
    -
        None


    """

    _val_keep(_keep)

    _val_X(_X)

    _val_do_not_drop(_do_not_drop, _X, _columns)

    _val_conflict(_conflict)

    _val_rtol(_rtol)

    _val_atol(_atol)

    _val_equal_nan(_equal_nan)

    _val_n_jobs(_n_jobs)






