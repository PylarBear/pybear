# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Callable, Literal, Sequence, Sequence
from typing_extensions import Union, TypeAlias
import numpy.typing as npt

import numbers
import numpy as np
import pandas as pd
import scipy.sparse as ss



DataType = Union[numbers.Real, str]

YContainer: TypeAlias = Union[npt.NDArray, pd.DataFrame, pd.Series, None]


SparseContainer: TypeAlias = Union[
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

XContainer: TypeAlias = Union[
    npt.NDArray,
    pd.DataFrame,
    SparseContainer
]

# the internal containers differ from the above external data containers
# by coo, dia, & bsr, because those cannot be sliced

InternalSparseContainer: TypeAlias = Union[
    ss._csr.csr_matrix,
    ss._csc.csc_matrix,
    ss._lil.lil_matrix,
    ss._dok.dok_matrix,
    ss._csr.csr_array,
    ss._csc.csc_array,
    ss._lil.lil_array,
    ss._dok.dok_array
]

InternalXContainer: TypeAlias = Union[
    npt.NDArray[DataType],
    pd.DataFrame,
    InternalSparseContainer
]


CountThresholdType: TypeAlias = \
    Union[numbers.Integral, Sequence[numbers.Integral]]

OriginalDtypesType: TypeAlias = npt.NDArray[
    Union[Literal['bin_int', 'int', 'float', 'obj']]
]

TotalCountsByColumnType: TypeAlias = dict[int, dict[any, int]]

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









