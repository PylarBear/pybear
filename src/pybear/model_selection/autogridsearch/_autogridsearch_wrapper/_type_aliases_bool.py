# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence, Tuple
from typing_extensions import TypeAlias, Union

import numbers



# see _type_aliases, bool subtypes of DataType, GridType, PointsType, ParamType
BoolDataType: TypeAlias = bool  # DataType sub
InBoolGridType: TypeAlias = Sequence[BoolDataType]
InBoolPointsType: TypeAlias = Union[None, numbers.Integral]
InBoolParamType: TypeAlias = Sequence[Tuple[InBoolGridType, InBoolPointsType, str]]
BoolGridType: TypeAlias = Sequence[BoolDataType]   # pizza finalize this, Sequence or list[BoolDataType]  # GridType sub
BoolPointsType: TypeAlias = int  # PointsType sub
BoolParamType: TypeAlias = list[BoolGridType, BoolPointsType, str] # ParamType sub








