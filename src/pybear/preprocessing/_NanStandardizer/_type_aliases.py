# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy.typing as npt
from typing_extensions import TypeAlias, Union

import pandas as pd
import polars as pl
import scipy.sparse as ss

import numbers




NumpyTypes: TypeAlias = npt.NDArray[numbers.Real]

PandasTypes: TypeAlias = Union[pd.DataFrame, pd.Series]

PolarsTypes: TypeAlias = Union[pl.DataFrame, pl.Series]

# dok and lil are left out intentionally
SparseTypes: TypeAlias = Union[
    ss._csr.csr_matrix, ss._csc.csc_matrix, ss._coo.coo_matrix,
    ss._dia.dia_matrix, ss._bsr.bsr_matrix, ss._csr.csr_array,
    ss._csc.csc_array, ss._coo.coo_array, ss._dia.dia_array,
    ss._bsr.bsr_array
]

XContainer: TypeAlias = Union[NumpyTypes, PandasTypes, PolarsTypes, SparseTypes]







