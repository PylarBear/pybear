# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing import Iterable, Literal, TypeAlias
from typing_extensions import Union
import numpy.typing as npt
import pandas as pd
import scipy.sparse as ss



ColumnsType: TypeAlias = Union[
    Iterable[str],
    None
]

ConflictType: TypeAlias = Literal['raise', 'ignore']

SparseTypes: TypeAlias = Union[
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

DataType: TypeAlias = Union[
    npt.NDArray,
    pd.DataFrame,
    SparseTypes
]

DoNotDropType: TypeAlias = Union[
    Iterable[str],
    Iterable[int],
    None
]

KeepType: TypeAlias = Union[
    Literal['first'],
    Literal['last'],
    Literal['random']
]


