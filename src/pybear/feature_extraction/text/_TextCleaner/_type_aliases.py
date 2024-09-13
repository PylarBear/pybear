# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing_extensions import Union, TypeAlias

import numpy.typing as npt

OuterIterableType: TypeAlias = Union[npt.NDArray, list]
DataType: TypeAlias = str
InnerIterableType: TypeAlias = Union[npt.NDArray[DataType], list[DataType], DataType]



CleanedTextType: TypeAlias = OuterIterableType[InnerIterableType]



















