# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from typing import TypeVar, Union, Iterable, Callable
from typing_extensions import TypeAlias
import numpy as np

# data is essentially all types that are not iterable
DataType = TypeVar('DataType', bool, int, float, str)
OriginalDtypesDtype: TypeAlias = np.ndarray[str]
TotalCountsByColumnType: TypeAlias = dict[int, dict[DataType, int]]
InstructionsType: TypeAlias = dict[int, list[Union[str, DataType]]]
IgnColsHandleAsBoolDtype: TypeAlias = \
    Union[Iterable[int], Iterable[str], Callable[[np.ndarray], np.ndarray], None]




