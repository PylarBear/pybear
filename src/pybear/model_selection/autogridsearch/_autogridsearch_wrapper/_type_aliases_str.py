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



# see _type_aliases, str subtypes for DataType, GridType, PointsType, ParamType
StrDataType: TypeAlias = Union[None, str]  # DataType sub

InStrGridType: TypeAlias = Sequence[StrDataType]
StrGridType: TypeAlias = list[StrDataType] # GridType sub

InPointsType: TypeAlias = Union[numbers.Integral, Sequence[numbers.Integral]]
PointsType: TypeAlias = list[numbers.Integral]

StrTypeType: TypeAlias = Literal['fixed_string']

InStrParamType: TypeAlias = Sequence[Tuple[InStrGridType, InPointsType, StrTypeType]]
StrParamType: TypeAlias = list[StrGridType, PointsType, StrTypeType] # ParamType sub






