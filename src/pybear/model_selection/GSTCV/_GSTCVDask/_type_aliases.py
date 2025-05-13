# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import ContextManager
from typing_extensions import (
    TypeAlias,
    Union
)

import dask
import distributed



# pizza figure out how to duck type these
DaskXType: TypeAlias = dask.array.core.Array
DaskYType: TypeAlias = Union[dask.array.core.Array, None]

DaskSlicerType: TypeAlias = dask.array.core.Array

DaskKFoldType: TypeAlias = tuple[DaskSlicerType, DaskSlicerType]

DaskSplitType: TypeAlias = tuple[DaskXType, DaskYType]

DaskSchedulerType: TypeAlias = Union[
    distributed.scheduler.Scheduler,
    distributed.client.Client,
    ContextManager  # nullcontext
]




