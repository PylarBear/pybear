# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import (
    Callable,
    Literal,
    TypedDict
)
from typing_extensions import (
    Any,
    Required,
    TypeAlias,
    Union
)
import numpy.typing as npt
from ..__shared._type_aliases import XContainer

import numbers

import pandas as pd
import scipy.sparse as ss
import polars as pl



# once any ss is inside partial_fit, inv_trfm, or transform it is
# converted to csc
InternalXContainer: TypeAlias = Union[
    npt.NDArray,
    pd.DataFrame,
    pl.DataFrame,
    ss.csc_matrix,
    ss.csc_array
]


KeepType: TypeAlias = Union[
    Literal['first', 'last', 'random', 'none'],
    dict[str, Any],
    numbers.Integral,
    str,
    Callable[[XContainer], int]
]


class InstructionType(TypedDict):

    keep: Required[Union[None, list[int]]]
    delete: Required[Union[None, list[int]]]
    add: Required[Union[None, dict[str, Any]]]


ConstantColumnsType: TypeAlias = dict[int, Any]

KeptColumnsType: TypeAlias = dict[int, Any]

RemovedColumnsType: TypeAlias = dict[int, Any]

ColumnMaskType: TypeAlias = npt.NDArray[bool]

NFeaturesInType: TypeAlias = int

FeatureNamesInType: TypeAlias = npt.NDArray[object]



