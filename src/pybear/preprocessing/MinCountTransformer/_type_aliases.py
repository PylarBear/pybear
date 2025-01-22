# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing import TypeVar, Iterable, Callable, Literal
from typing_extensions import Union, TypeAlias
import numpy.typing as npt



# pizza, probably get rid of this
# data is essentially all types that are not iterable
DataType = TypeVar('DataType', bool, int, float, str)

# pizza finalize these containers!
XContainer: TypeAlias = Iterable[Iterable[DataType]]

YContainer: TypeAlias = Union[Iterable[Iterable[DataType]], Iterable[DataType], None]

OriginalDtypesDtype: TypeAlias = npt.NDArray[
    Union[Literal['bin_int', 'int', 'float', 'obj']]
]

TotalCountsByColumnType: TypeAlias = dict[int, dict[DataType, int]]

InstructionsType: TypeAlias = dict[int, list[Union[str, DataType]]]

IgnoreColumnsType: TypeAlias = \
    Union[
        Iterable[int],
        Iterable[str],
        Callable[[XContainer], npt.NDArray],
        None
    ]

HandleAsBoolType: TypeAlias = \
    Union[
        Iterable[int],
        Iterable[str],
        Callable[[XContainer], npt.NDArray],
        None
    ]

InternalIgnoreColumnsType: Iterable[int]   # pizza, is this NDArray?

InternalHandleAsBoolType: TypeAlias = Iterable[int]   # pizza, is this NDArray?







