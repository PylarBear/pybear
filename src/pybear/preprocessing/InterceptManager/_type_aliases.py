# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Iterable, Literal, Optional, TypedDict, Callable
from typing_extensions import Union, TypeAlias, Required
import numpy.typing as npt
import pandas as pd
import scipy.sparse as ss



# pizzam make a decision, keep this here or just be explicit in every function?
ColumnsType: TypeAlias = Union[
    Iterable[str],
    None
]

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

KeepType: TypeAlias = Optional[Union[
    Literal['first', 'last', 'random', 'none'],
    dict[str, any],
    int,
    str,
    Callable[[DataType], int]
]]




class InstructionType(TypedDict):

    keep: Required[Union[None, list, npt.NDArray[int]]]
    delete: Required[Union[None, list, npt.NDArray[int]]]
    add: Required[Union[None, dict[str, any]]]









