# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Iterable
from typing_extensions import (
    TypeAlias,
    Union
)
import numpy.typing as npt


# pizza figure out how to duck type these
XInputType: TypeAlias = Iterable
XSKWIPType: TypeAlias = Iterable
# pizza figure out how to duck type these
YInputType: TypeAlias = Union[Iterable, None]
YSKWIPType: TypeAlias = Union[Iterable, None]

SKSlicerType: TypeAlias = npt.NDArray[int]

SKKFoldType: TypeAlias = tuple[SKSlicerType, SKSlicerType]




