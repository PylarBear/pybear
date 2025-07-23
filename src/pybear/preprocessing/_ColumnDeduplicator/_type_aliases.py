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
import numpy.typing as npt

import pandas as pd
import polars as pl
import scipy.sparse as ss



InternalXContainer: TypeAlias = Union[
    npt.NDArray,
    pd.DataFrame,
    pl.DataFrame,
    ss.csc_array,
    ss.csc_matrix
]

KeepType: TypeAlias = Literal['first', 'last', 'random']

DoNotDropType: TypeAlias = Union[Sequence[int], Sequence[str], None]

ConflictType: TypeAlias = Literal['raise', 'ignore']

DuplicatesType: TypeAlias = list[list[int]]

RemovedColumnsType: TypeAlias = dict[int, int]

ColumnMaskType: TypeAlias = npt.NDArray[bool]

FeatureNamesInType: TypeAlias = npt.NDArray[str]




