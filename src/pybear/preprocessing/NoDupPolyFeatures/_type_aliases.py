# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Iterable, Literal
from typing_extensions import Union, TypeAlias
import numpy.typing as npt
import pandas as pd
import scipy.sparse as ss






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




DuplicatesType: TypeAlias = list[list[tuple[int]]]
#         list[list[int]] - a list of the groups of identical
#         columns, indicated by their zero-based column index positions
#         in the originally fit data.
#
DroppedDuplicatesType: TypeAlias = dict[tuple[int], tuple[int]]
#         dict[int, int] - a dictionary whose keys are the indices of
#         duplicate columns removed from the original data, indexed by
#         their column location in the original data; the values are the
#         column index in the original data of the respective duplicate
#         that was kept.
#
#     column_mask_:
#         list[bool], shape (n_features_,) - Indicates which
#         columns of the fitted data are kept (True) and which are removed
#         (False) during transform.
#
ConstantsType: TypeAlias = dict[tuple[int], any]
#         put words about how the only constant, in this unforgiving world, is good pizza.
#
DroppedConstantsType: TypeAlias = dict[tuple[int], any]
#
#
#







