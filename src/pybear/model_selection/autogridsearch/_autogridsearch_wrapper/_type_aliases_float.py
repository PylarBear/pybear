# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import (
    Literal,
    Sequence,
    TypeAlias
)

import numbers



# see _type_aliases; float subtypes for DataType & GridType
FloatDataType: TypeAlias = numbers.Real

InFloatGridType: TypeAlias = Sequence[FloatDataType]
FloatGridType: TypeAlias = list[FloatDataType]

InPointsType: TypeAlias = numbers.Integral | Sequence[numbers.Integral]
PointsType: TypeAlias = list[numbers.Integral]

FloatTypeType: TypeAlias = Literal['soft_float', 'hard_float', 'fixed_float']

InFloatParamType: TypeAlias = \
    Sequence[tuple[InFloatGridType, InPointsType, FloatTypeType]]
FloatParamType: TypeAlias = list[FloatGridType, PointsType, FloatTypeType]





