# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import (
    Iterable,
    Sequence,
    Literal,
    Callable,
    TypedDict,
    Protocol
)
from typing_extensions import (
    Any,
    Self,
    TypeAlias,
    Union,
    NotRequired
)
import numpy.typing as npt

import numbers

import numpy as np



CVResultsType: TypeAlias = \
    dict[str, np.ma.masked_array[Union[float, dict[str, Any]]]]

IntermediateHolderType: TypeAlias = Union[
    np.ma.masked_array[float],
    npt.NDArray[numbers.Real]
]

ParamGridType: TypeAlias = dict[str, Sequence[Any]]

ParamGridsType: TypeAlias = Sequence[ParamGridType]

GenericSlicerType: TypeAlias = Sequence[numbers.Integral]

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

# pizza finalize this
ScorerCallableType: TypeAlias = Callable[[Iterable, Iterable, ...], numbers.Real]


ScorerInputType: TypeAlias = Union[
    ScorerNameTypes,
    Sequence[ScorerNameTypes],
    ScorerCallableType,
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


RefitCallableType: TypeAlias = Callable[[CVResultsType], numbers.Integral]
RefitType: TypeAlias = Union[bool, ScorerNameTypes, RefitCallableType]


# pizza what about a ThresholdsType?


class ClassifierProtocol(Protocol):

    def fit(self, X: Any, y: Any) -> Self:
        ...

    # The default 'score' method of the estimator can never be used, as
    # the decision threshold cannot be manipulated. Therefore, it is not
    # necessary for the estimator to have a 'score' method.
    # def score(self, y_pred: Any, y_act: Any) -> Any:
    #     ...

    def get_params(self, *args, **kwargs) -> dict[str, Any]:
        ...

    def set_params(self, *args, **kwargs) -> Self:
        ...

    def predict_proba(self, *args, **kwargs) -> Any:
        ...



