# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from typing import TypeAlias, TypeVar
import numpy as np

# data is essentially all types that are not iterable
DataType = TypeVar('DataType', bool, int, float, str, type(None))

OriginalDtypesDtype: TypeAlias = np.ndarray[str]
TotalCountsByColumnType: TypeAlias = dict[int, dict[DataType, int]]
InstructionsType: TypeAlias = dict[int]  # pizza fix this





