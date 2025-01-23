# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import TypeVar, Iterable, Callable, Literal
from typing_extensions import Union, TypeAlias
import numpy.typing as npt

import numbers
import numpy as np



# pizza, probably get rid of this
# data is essentially all types that are not iterable
DataType = TypeVar('DataType', bool, int, float, str)

# pizza finalize these containers!
XContainer: TypeAlias = Iterable[Iterable[DataType]]

YContainer: TypeAlias = Union[Iterable[Iterable[DataType]], Iterable[DataType], None]

# pizza remember to proliferate new type
CountThresholdType: TypeAlias = \
    Union[numbers.Integral, Iterable[numbers.Integral]]

OriginalDtypesType: TypeAlias = npt.NDArray[
    Union[Literal['bin_int', 'int', 'float', 'obj']]
]

TotalCountsByColumnType: TypeAlias = dict[int, dict[DataType, int]]

InstructionsType: TypeAlias = dict[int, list[Union[str, DataType]]]

IgnoreColumnsType: TypeAlias = \
    Union[
        None,
        Iterable[int],
        Iterable[str],
        Callable[[XContainer], Union[Iterable[int], Iterable[str]]]
    ]

HandleAsBoolType: TypeAlias = \
    Union[
        None,
        Iterable[int],
        Iterable[str],
        Callable[[XContainer], Union[Iterable[int], Iterable[str]]]
    ]

InternalIgnoreColumnsType: TypeAlias = npt.NDArray[np.int32]

InternalHandleAsBoolType: TypeAlias = npt.NDArray[np.int32]







