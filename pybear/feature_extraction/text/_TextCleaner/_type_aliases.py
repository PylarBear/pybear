# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing import Iterable, Union, TypeAlias

import numpy as np



OuterIterableType: TypeAlias = Union[np.ndarray, list]
DataType: TypeAlias = str
InnerIterableType: TypeAlias = Union[np.ndarray[DataType], list[DataType], DataType]



CleanedTextType: TypeAlias = OuterIterableType[InnerIterableType]



















