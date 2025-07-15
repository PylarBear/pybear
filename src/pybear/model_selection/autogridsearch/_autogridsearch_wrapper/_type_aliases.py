# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import (
    Literal,
    Sequence
)
from typing_extensions import (
    TypeAlias,
    Union
)
from ._type_aliases_float import (
    FloatDataType,
    InFloatGridType,
    FloatGridType,
    InFloatParamType,
    FloatParamType,
)
from ._type_aliases_int import (
    IntDataType,
    InIntGridType,
    IntGridType,
    InIntParamType,
    IntParamType
)
from ._type_aliases_str import (
    StrDataType,
    InStrGridType,
    StrGridType,
    InStrParamType,
    StrParamType
)
from ._type_aliases_bool import (
    BoolDataType,
    InBoolGridType,
    BoolGridType,
    InBoolParamType,
    BoolParamType
)

import numbers



DataType: TypeAlias = Union[BoolDataType, StrDataType, IntDataType, FloatDataType]

InGridType: TypeAlias = \
    Union[InBoolGridType, InStrGridType, InIntGridType, InFloatGridType]
GridType: TypeAlias = Union[BoolGridType, StrGridType, IntGridType, FloatGridType]

InPointsType: TypeAlias = Union[numbers.Integral, Sequence[numbers.Integral]]
PointsType: TypeAlias = list[numbers.Integral]

InParamType: TypeAlias = \
    Union[InBoolParamType, InStrParamType, InIntParamType, InFloatParamType]
ParamType: TypeAlias = \
    Union[BoolParamType, StrParamType, IntParamType, FloatParamType]

InParamsType: TypeAlias = dict[
    str,
    Union[InBoolParamType, InStrParamType, InIntParamType, InFloatParamType]
]
ParamsType: TypeAlias = dict[
    str,
    Union[BoolParamType, StrParamType, IntParamType, FloatParamType]
]

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

LogspaceType: TypeAlias = Union[Literal[False], numbers.Integral]
IsLogspaceType: TypeAlias = dict[str, LogspaceType]

PhliteType: TypeAlias = dict[str, bool]

ParamGridType: TypeAlias = dict[str, GridType]

GridsType: TypeAlias = dict[int, ParamGridType]

BestParamsType: TypeAlias = dict[str, DataType]

ResultsType: TypeAlias = dict[int, BestParamsType]






