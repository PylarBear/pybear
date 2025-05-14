# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import (
    ContextManager,
    Iterable,
    Sequence
)
from typing_extensions import (
    TypeAlias,
    Union
)

import numbers

import dask
import distributed



DaskXType: TypeAlias = Iterable
DaskYType: TypeAlias = Union[Sequence[numbers.Integral], None]

DaskSlicerType: TypeAlias = dask.array.core.Array

DaskKFoldType: TypeAlias = tuple[DaskSlicerType, DaskSlicerType]

DaskSplitType: TypeAlias = tuple[DaskXType, DaskYType]

DaskSchedulerType: TypeAlias = Union[
    distributed.scheduler.Scheduler,
    distributed.client.Client,
    ContextManager  # nullcontext
]




