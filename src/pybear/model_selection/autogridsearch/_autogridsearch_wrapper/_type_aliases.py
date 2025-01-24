# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import TypeVar
from typing_extensions import Union, TypeAlias



DataType = TypeVar('DataType', int, float, bool, str)

GridType: TypeAlias = Union[list[DataType], tuple[DataType], set[DataType]]

PointsType: TypeAlias = Union[int, Union[list[int], tuple[int]]]

ParamType: TypeAlias = list[GridType, PointsType, str]

ParamsType: TypeAlias = dict[str, ParamType]

GridsType: TypeAlias = dict[int, dict[str, GridType]]

BestParamsType: TypeAlias = dict[str, DataType]

ResultsType: TypeAlias = dict[int, BestParamsType]























