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



SKXType: TypeAlias = Iterable
SKYType: TypeAlias = Union[Sequence[numbers.Integral], None]

SKSlicerType: TypeAlias = Sequence[numbers.Integral]

SKKFoldType: TypeAlias = tuple[SKSlicerType, SKSlicerType]

SKSplitType: TypeAlias = tuple[SKXType, SKYType]

SKSchedulerType: TypeAlias = ContextManager    # nullcontext




