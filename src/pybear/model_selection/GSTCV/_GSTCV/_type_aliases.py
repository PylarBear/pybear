# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import (
    ContextManager,
    Iterable,
    Literal,
    Sequence,
    TypeAlias
)

import numbers


PreDispatchType: TypeAlias = Literal['all'] | str | numbers.Integral

SKXType: TypeAlias = Iterable
SKYType: TypeAlias = Sequence[numbers.Integral] | None

SKSlicerType: TypeAlias = Sequence[numbers.Integral]

SKKFoldType: TypeAlias = tuple[SKSlicerType, SKSlicerType]

SKSplitType: TypeAlias = tuple[SKXType, SKYType]

SKSchedulerType: TypeAlias = ContextManager    # nullcontext




