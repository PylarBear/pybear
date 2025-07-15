# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import (
    Literal,
    Sequence,
    Tuple
)
from typing_extensions import (
    TypeAlias,
    Union
)

import numbers



# see _type_aliases, bool subtypes of DataType, GridType, PointsType, ParamType
BoolDataType: TypeAlias = Union[bool, None]  # DataType sub

InBoolGridType: TypeAlias = Sequence[BoolDataType]
BoolGridType: TypeAlias = list[BoolDataType]

InPointsType: TypeAlias = Union[numbers.Integral, Sequence[numbers.Integral]]
PointsType: TypeAlias = list[numbers.Integral]

BoolTypeType: TypeAlias = Literal['fixed_bool']

InBoolParamType: TypeAlias = \
    Sequence[Tuple[InBoolGridType, InPointsType, BoolTypeType]]
BoolParamType: TypeAlias = list[BoolGridType, PointsType, BoolTypeType] # ParamType sub








