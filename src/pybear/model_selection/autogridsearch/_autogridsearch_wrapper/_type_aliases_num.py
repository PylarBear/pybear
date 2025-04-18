# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence, Tuple
from typing_extensions import (
    TypeAlias,
    Union
)
from ._type_aliases import PointsType
from ._type_aliases_float import FloatDataType
from ._type_aliases_int import IntDataType



# see _type_aliases, general num subtypes of DataType, GridType, PointsType, ParamType
NumDataType: TypeAlias = Union[IntDataType, FloatDataType]
InNumGridType: TypeAlias = Sequence[NumDataType]
InNumParamType: TypeAlias = Sequence[Tuple[InNumGridType, PointsType, str]]
NumGridType: TypeAlias = list[NumDataType]
NumPointsType: TypeAlias = list[int]
NumParamType: TypeAlias = list[NumGridType, PointsType, str]





