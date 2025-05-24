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

import numbers

import pandas as pd
import scipy.sparse as ss
import polars as pl



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


# once any ss is inside partial_fit, inv_trfm, or transform it is converted
# to csc
InternalDataContainer: TypeAlias = Union[
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
    Callable[[DataContainer], int]
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



