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

DataType: TypeAlias = Union[int, float, np.float64]

XInputType: TypeAlias = Iterable[Iterable[DataType]]
XSKWIPType: TypeAlias = npt.NDArray[DataType]
XDaskWIPType: TypeAlias = da.core.Array

YInputType: TypeAlias = Union[Iterable[Iterable[DataType]], Iterable[DataType], None]
YSKWIPType: TypeAlias = Union[npt.NDArray[DataType], None]
YDaskWIPType: TypeAlias = Union[da.core.Array, None]

CVResultsType: TypeAlias = \
    dict[str, np.ma.masked_array[Union[float, dict[str, any]]]]

IntermediateHolderType: TypeAlias = Union[
    np.ma.masked_array[float],
    npt.NDArray[Union[int, float]]
]

ParamGridType: TypeAlias = dict[str, Union[list[any], npt.NDArray[any]]]

SKKFoldType: TypeAlias = npt.NDArray[int]
DaskKFoldType: TypeAlias = da.core.Array
GenericKFoldType: TypeAlias = Iterable[tuple[Iterable[int], Iterable[int]]]

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
RefitType: TypeAlias = Union[bool, None, ScorerNameTypes, RefitCallableType]







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








