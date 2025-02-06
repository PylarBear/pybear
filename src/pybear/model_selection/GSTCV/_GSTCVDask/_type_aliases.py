# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Iterable, Literal, Callable, TypedDict, Protocol
from typing_extensions import TypeAlias, Union, NotRequired

import numpy as np
import numpy.typing as npt

import dask
import distributed




DataType: TypeAlias = Union[int, float, np.float64]

XInputType: TypeAlias = Iterable[Iterable[DataType]]
XSKWIPType: TypeAlias = npt.NDArray[DataType]
XDaskWIPType: TypeAlias = dask.array.core.Array

YInputType: TypeAlias = Union[Iterable[Iterable[DataType]], Iterable[DataType], None]
YSKWIPType: TypeAlias = Union[npt.NDArray[DataType], None]
YDaskWIPType: TypeAlias = Union[dask.array.core.Array, None]

CVResultsType: TypeAlias = \
    dict[str, np.ma.masked_array[Union[float, dict[str, any]]]]

IntermediateHolderType: TypeAlias = Union[
    np.ma.masked_array[float],
    npt.NDArray[Union[int, float]]
]

ParamGridType: TypeAlias = Union[
    dict[str, Union[list[any], npt.NDArray[any]]],
    list[dict[str, Union[list[any], npt.NDArray[any]]]]
]
SKSlicerType: TypeAlias = npt.NDArray[int]
DaskSlicerType: TypeAlias = dask.array.core.Array
GenericSlicerType: TypeAlias = Iterable[int]

SKKFoldType: TypeAlias = tuple[SKSlicerType, SKSlicerType]
DaskKFoldType: TypeAlias = tuple[DaskSlicerType, DaskSlicerType]
GenericKFoldType: TypeAlias = tuple[GenericSlicerType, GenericSlicerType]

FeatureNamesInType: TypeAlias = Union[npt.NDArray[str], None]

# scoring / scorer ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
ScorerNameTypes: TypeAlias = Literal[
    'accuracy',
    'balanced_accuracy',
    'average_precision',
    'f1',
    'precision',
    'recall'
]

ScorerCallableType: TypeAlias = Callable[[YInputType, YInputType, ...], float]


ScorerInputType: TypeAlias = Union[
    ScorerNameTypes,
    ScorerCallableType,
    list[ScorerNameTypes],
    dict[str, ScorerCallableType]
]


class ScorerWIPType(TypedDict):

    accuracy: NotRequired[ScorerCallableType]
    balanced_accuracy: NotRequired[ScorerCallableType]
    average_precision: NotRequired[ScorerCallableType]
    f1: NotRequired[ScorerCallableType]
    precision: NotRequired[ScorerCallableType]
    recall: NotRequired[ScorerCallableType]
    score: NotRequired[ScorerCallableType]
# END scoring / scorer ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * *


RefitCallableType: TypeAlias = Callable[[CVResultsType], int]
RefitType: TypeAlias = Union[bool, ScorerNameTypes, RefitCallableType, None]


SchedulerType: TypeAlias = Union[
    distributed.scheduler.Scheduler,
    distributed.client.Client
]


class ClassifierProtocol(Protocol):

    def fit(self, X: any, y: any) -> any:
        ...

    # The default 'score' method of the estimator can never be used, as
    # the decision threshold cannot be manipulated. Therefore, it is not
    # necessary for the estimator to have a 'score' method.
    # def score(self, y_pred: any, y_act: any) -> any:
    #     ...

    def get_params(self, *args, **kwargs) -> any:
        ...

    def set_params(self, *args, **kwargs) -> any:
        ...

    def predict_proba(self, *args, **kwargs) -> any:
        ...








