# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import (
    Callable,
    Literal,
    Sequence
)
from typing_extensions import (
    TypeAlias,
    Union
)
import numpy.typing as npt
from ..__shared._type_aliases import XContainer

import numbers

import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as ss



DataType = Union[numbers.Number, str]

YContainer: TypeAlias = Union[
    npt.NDArray,
    pd.DataFrame,
    pd.Series,
    pl.DataFrame,
    pl.Series,
    None
]

InternalXContainer: TypeAlias = Union[
    npt.NDArray[DataType],
    pd.DataFrame,
    pl.DataFrame,
    ss.csc_array,
    ss.csc_matrix
]


CountThresholdType: TypeAlias = \
    Union[numbers.Integral, Sequence[numbers.Integral]]

OriginalDtypesType: TypeAlias = npt.NDArray[
    Union[Literal['bin_int', 'int', 'float', 'obj']]
]

TotalCountsByColumnType: TypeAlias = dict[int, dict[DataType, int]]

InstructionsType: TypeAlias = \
    dict[
        int,
        list[Union[DataType, Literal['INACTIVE', 'DELETE ALL', 'DELETE COLUMN']]]
    ]

IgnoreColumnsType: TypeAlias = \
    Union[
        None,
        Sequence[numbers.Integral],
        npt.NDArray[np.int32],
        Sequence[str],
        Callable[[XContainer], Union[Sequence[numbers.Integral], Sequence[str]]]
    ]

HandleAsBoolType: TypeAlias = \
    Union[
        None,
        Sequence[numbers.Integral],
        npt.NDArray[np.int32],
        Sequence[str],
        Callable[[XContainer], Union[Sequence[numbers.Integral], Sequence[str]]]
    ]

InternalIgnoreColumnsType: TypeAlias = npt.NDArray[np.int32]

InternalHandleAsBoolType: TypeAlias = npt.NDArray[np.int32]

FeatureNamesInType: TypeAlias = npt.NDArray[object]







