# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import (
    ContextManager,
    Iterable,
    Literal,
    Optional,
    Sequence
)
from typing_extensions import (
    TypeAlias,
    Union
)

import numbers


PreDispatchType: TypeAlias = Optional[Union[Literal['all'], str, numbers.Integral]]

SKXType: TypeAlias = Iterable
SKYType: TypeAlias = Union[Sequence[numbers.Integral], None]

SKSlicerType: TypeAlias = Sequence[numbers.Integral]

SKKFoldType: TypeAlias = tuple[SKSlicerType, SKSlicerType]

SKSplitType: TypeAlias = tuple[SKXType, SKYType]

SKSchedulerType: TypeAlias = ContextManager    # nullcontext




