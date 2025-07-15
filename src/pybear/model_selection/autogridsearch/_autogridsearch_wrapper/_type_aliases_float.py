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



# see _type_aliases; float subtypes for DataType & GridType
FloatDataType: TypeAlias = numbers.Real

InFloatGridType: TypeAlias = Sequence[FloatDataType]
FloatGridType: TypeAlias = list[FloatDataType]

InPointsType: TypeAlias = Union[numbers.Integral, Sequence[numbers.Integral]]
PointsType: TypeAlias = list[numbers.Integral]

FloatTypeType: TypeAlias = Literal['soft_float', 'hard_float', 'fixed_float']

InFloatParamType: TypeAlias = \
    Sequence[Tuple[InFloatGridType, InPointsType, FloatTypeType]]
FloatParamType: TypeAlias = list[FloatGridType, PointsType, FloatTypeType]





