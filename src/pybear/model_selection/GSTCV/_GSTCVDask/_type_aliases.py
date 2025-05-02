# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import (
    TypeAlias,
    Union
)

import numbers

import dask
import distributed



XDaskInputType: TypeAlias = dask.array.core.Array
XDaskWIPType: TypeAlias = dask.array.core.Array

YDaskInputType: TypeAlias = Union[dask.array.core.Array, None]
YDaskWIPType: TypeAlias = Union[dask.array.core.Array, None]

DaskSlicerType: TypeAlias = dask.array.core.Array

DaskKFoldType: TypeAlias = tuple[DaskSlicerType, DaskSlicerType]

SchedulerType: TypeAlias = Union[
    distributed.scheduler.Scheduler,
    distributed.client.Client
]




