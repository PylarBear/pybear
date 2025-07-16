# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import (
    Sequence,
    Tuple
)
from typing_extensions import (
    TypeAlias,
    TypeVar,
    Union
)
from ._type_aliases_float import (
    FloatDataType,
    InFloatGridType,
    FloatGridType,
    FloatTypeType
)
from ._type_aliases_int import (
    IntDataType,
    InIntGridType,
    IntGridType,
    IntTypeType
)

import numbers



# see _type_aliases, general num subtypes of DataType, GridType, PointsType, ParamType
NumDataType = TypeVar('NumDataType', IntDataType, FloatDataType)

InNumGridType: TypeAlias = Union[InIntGridType, InFloatGridType]
NumGridType: TypeAlias = Union[IntGridType, FloatGridType]

InPointsType: TypeAlias = Union[numbers.Integral, Sequence[numbers.Integral]]
PointsType: TypeAlias = list[numbers.Integral]

NumTypeType: TypeAlias = Union[IntTypeType, FloatTypeType]

InNumParamType: TypeAlias = Sequence[Tuple[InNumGridType, InPointsType, NumTypeType]]
NumParamType: TypeAlias = list[NumGridType, PointsType, NumTypeType]





