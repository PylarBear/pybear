# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import (
    Protocol,
    Sequence,
    TypeVar
)
from typing_extensions import (
    Any,
    Self,
    TypeAlias,
    Union
)

import numbers



DataType = TypeVar('DataType', numbers.Real, str)

GridType: TypeAlias = Sequence[DataType]

PointsType: TypeAlias = Union[int, Sequence[int]]

ParamType: TypeAlias = list[GridType, PointsType, str]

ParamsType: TypeAlias = dict[str, ParamType]

GridsType: TypeAlias = dict[int, dict[str, GridType]]

BestParamsType: TypeAlias = dict[str, DataType]

ResultsType: TypeAlias = dict[int, BestParamsType]




class EstimatorProtocol(Protocol):

    def fit(self, X: any, y: any) -> Self:
        ...

    def get_params(self, *args, **kwargs) -> dict[str, Any]:
        ...

    def set_params(self, *args, **kwargs) -> Self:
        ...

    # pizza what about score?
















