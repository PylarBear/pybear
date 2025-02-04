# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import TypeVar, Sequence
from typing_extensions import Union, TypeAlias

import numbers


DataType = TypeVar('DataType', numbers.Real, str)

GridType: TypeAlias = Sequence[DataType]

PointsType: TypeAlias = Union[int, Sequence[int]]

ParamType: TypeAlias = list[GridType, PointsType, str]

ParamsType: TypeAlias = dict[str, ParamType]

GridsType: TypeAlias = dict[int, dict[str, GridType]]

BestParamsType: TypeAlias = dict[str, DataType]

ResultsType: TypeAlias = dict[int, BestParamsType]























