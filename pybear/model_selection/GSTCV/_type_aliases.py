# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import (TypeAlias, Union, Iterable, Literal, Callable, TypedDict,
                    NotRequired, Protocol)

import numpy as np
import numpy.typing as npt

import dask.array as da
import distributed





SchedulerType: TypeAlias = distributed.scheduler.Scheduler

DataType: TypeAlias = Union[int, float]

XInputType: TypeAlias = Iterable[Iterable[DataType]]
XSKWIPType: TypeAlias = npt.NDArray[DataType]
XDaskWIPType: TypeAlias = da.core.Array

YInputType: TypeAlias = Union[Iterable[Iterable[DataType]], Iterable[DataType], None]
YSKWIPType: TypeAlias = Union[npt.NDArray[DataType], None]
YDaskWIPType: TypeAlias = Union[da.core.Array, None]

CVResultsType: TypeAlias = \
    dict[str, np.ma.masked_array[Union[float, dict[str, any]]]]
IntermediateHolderType: TypeAlias = Union[np.ma.masked_array[float], npt.NDArray[Union[int, float]]]
ParamGridType: TypeAlias = dict[str, Union[list[any], npt.NDArray[any]]]

SKKFoldType: TypeAlias = npt.NDArray[int]
DaskKFoldType: TypeAlias = da.core.Array

FeatureNamesInType: TypeAlias = Union[npt.NDArray[str], None]


ScorerNameTypes: TypeAlias = Literal[
    'accuracy',
    'balanced_accuracy',
    'average_precision',
    'f1',
    'precision',
    'recall'
]

ScorerCallableType: TypeAlias = Callable[[YInputType, YInputType, ...], np.float64]

ScorerInputType: TypeAlias = Union[None, ScorerNameTypes, list[ScorerNameTypes]]


class ScorerWIPType(TypedDict):

    accuracy: NotRequired[ScorerCallableType]
    balanced_accuracy: NotRequired[ScorerCallableType]
    average_precision: NotRequired[ScorerCallableType]
    f1: NotRequired[ScorerCallableType]
    precision: NotRequired[ScorerCallableType]
    recall: NotRequired[ScorerCallableType]


class ClassifierProtocol(Protocol):

    def fit(self, X: any, y: any) -> any:
        ...

    def score(self, y_pred: any, y_act: any) -> any:
        ...

    def get_params(self, *args, **kwargs) -> any:
        ...

    def set_params(self, *args, **kwargs) -> any:
        ...

    def predict_proba(self, *args, **kwargs) -> any:
        ...








