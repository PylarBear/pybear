# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Iterable
from typing_extensions import Union, TypeAlias

import numpy as np
import dask.array as da


OuterIterableType: TypeAlias = Iterable #Union[list, np.ndarray, da.core.Array]
DataType: TypeAlias = str
InnerIterableType: TypeAlias = Union[np.ndarray[DataType], list[DataType], DataType]



CleanedTextType: TypeAlias = OuterIterableType[InnerIterableType]


ListOfListsType: TypeAlias = Iterable[Iterable[str]]
ListOfStringsType: TypeAlias = Iterable[str]
















