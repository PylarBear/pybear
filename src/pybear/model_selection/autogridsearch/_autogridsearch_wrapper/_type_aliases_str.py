# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence, Tuple
from typing_extensions import TypeAlias, Union

import numbers



# see _type_aliases, str subtypes for DataType, GridType, PointsType, ParamType
StrDataType: TypeAlias = Union[None, str]  # DataType sub --- pizza verify about the None
InStrGridType: TypeAlias = Sequence[StrDataType]
InStrPointsType: TypeAlias = Union[None, numbers.Integral]
InStrParamType: TypeAlias = Sequence[Tuple[InStrGridType, InStrPointsType, str]]
StrGridType: TypeAlias = Sequence[StrDataType] # GridType sub --- pizza do we want list[StrDataType]?
StrPointsType: TypeAlias = numbers.Integral # PointsType sub
StrParamType: TypeAlias = list[StrGridType, StrPointsType, str] # ParamType sub






