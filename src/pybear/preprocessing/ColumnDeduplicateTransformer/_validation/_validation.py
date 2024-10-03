# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from ._do_not_drop import _val_do_not_drop
from ._keep import _val_keep
from ._columns import _val_columns
from ._n_jobs import _val_n_jobs

from .._type_aliases import (
    DataType,
    KeepType,
    DoNotDropType,
    ColumnsType
)
from typing_extensions import Union



def _validation(
    _X:DataType,
    _keep:KeepType,
    _do_not_drop:DoNotDropType,
    _columns: ColumnsType,
    _n_jobs: Union[int, None]
) -> None:

    _val_columns(_X, _columns)

    _val_keep(_keep)

    _val_do_not_drop(_do_not_drop, _X, _columns)

    _val_n_jobs(_n_jobs)








