# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Iterable, Sequence
from typing_extensions import (
    TypeAlias,
    Union
)

import numbers



# pizza figure out how to duck type these
XSKInputType: TypeAlias = Iterable
XSKWIPType: TypeAlias = Iterable
# pizza figure out how to duck type these
YSKInputType: TypeAlias = Union[Sequence[numbers.Integral], None]
YSKWIPType: TypeAlias = Union[Sequence[numbers.Integral], None]

SKSlicerType: TypeAlias = Iterable[numbers.Integral]

SKKFoldType: TypeAlias = tuple[SKSlicerType, SKSlicerType]




