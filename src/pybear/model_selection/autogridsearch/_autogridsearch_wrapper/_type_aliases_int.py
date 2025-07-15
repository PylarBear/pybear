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



# see _type_aliases - int subtypes for DataType, GridType
IntDataType: TypeAlias = numbers.Integral

InIntGridType: TypeAlias = Sequence[IntDataType]
IntGridType: TypeAlias = list[IntDataType]

InPointsType: TypeAlias = Union[numbers.Integral, Sequence[numbers.Integral]]
PointsType: TypeAlias = list[numbers.Integral]

IntTypeType: TypeAlias = Literal['soft_integer', 'hard_integer', 'fixed_integer']

InIntParamType: TypeAlias = \
    Sequence[Tuple[InIntGridType, InPointsType, IntTypeType]]
IntParamType: TypeAlias = list[IntGridType, PointsType, IntTypeType]








