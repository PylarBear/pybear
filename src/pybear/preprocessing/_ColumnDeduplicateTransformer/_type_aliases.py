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



DataContainer: TypeAlias = Union[
    npt.NDArray,
    pd.DataFrame,
    pl.DataFrame,
    ss._csr.csr_matrix,
    ss._csc.csc_matrix,
    ss._coo.coo_matrix,
    ss._dia.dia_matrix,
    ss._lil.lil_matrix,
    ss._dok.dok_matrix,
    ss._bsr.bsr_matrix,
    ss._csr.csr_array,
    ss._csc.csc_array,
    ss._coo.coo_array,
    ss._dia.dia_array,
    ss._lil.lil_array,
    ss._dok.dok_array,
    ss._bsr.bsr_array
]

InternalDataContainer: TypeAlias = Union[
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




