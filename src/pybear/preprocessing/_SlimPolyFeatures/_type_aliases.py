# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence, Literal, Callable
from typing_extensions import Union, TypeAlias
import numpy.typing as npt
import pandas as pd
import scipy.sparse as ss






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

DataContainer: TypeAlias = Union[
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

# pizza polars
InternalDataContainer: TypeAlias = Union[
    npt.NDArray,
    pd.DataFrame,
    InternalSparseContainer
]


FeatureNameCombinerType: TypeAlias = \
    Union[
        Callable[[Sequence[str], tuple[int, ...]], str],
        Literal['as_feature_names', 'as_indices']
    ]

ExpansionCombinationsType: tuple[tuple[int, ...], ...]

PolyDuplicatesType: TypeAlias = list[list[tuple[int, ...]]]

KeptPolyDuplicatesType: TypeAlias = dict[tuple[int, ...], list[tuple[int, ...]]]

DroppedPolyDuplicatesType: TypeAlias = dict[tuple[int, ...], tuple[int, ...]]

PolyConstantsType: TypeAlias = dict[tuple[int, ...], any]





