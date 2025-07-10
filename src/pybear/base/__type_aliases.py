# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import (
    TypeAlias,
    Union
)
import numpy.typing as npt

import pandas as pd
import polars as pl
import scipy.sparse as ss



Python1DTypes: TypeAlias = Union[list, tuple, set]
Python2DTypes: TypeAlias = Union[list[list], tuple[tuple]]
PythonTypes: TypeAlias = Union[list, tuple, set, list[list], tuple[tuple]]

Numpy1DTypes: TypeAlias = npt.NDArray
Numpy2DTypes: TypeAlias = npt.NDArray
NumpyTypes: TypeAlias = npt.NDArray

Pandas1DTypes: TypeAlias = pd.core.series.Series
Pandas2DTypes: TypeAlias = pd.core.frame.DataFrame
PandasTypes: TypeAlias = Union[pd.core.series.Series, pd.core.frame.DataFrame]

Polars1DTypes: TypeAlias = pl.series.Series
Polars2DTypes: TypeAlias = pl.dataframe.DataFrame
PolarsTypes: TypeAlias = Union[pl.series.Series, pl.dataframe.DataFrame]

ScipySparseTypes: TypeAlias = Union[
    ss.csc_matrix, ss.csc_array, ss.csr_matrix, ss.csr_array,
    ss.coo_matrix, ss.coo_array, ss.dia_matrix, ss.dia_array,
    ss.lil_matrix, ss.lil_array, ss.dok_matrix, ss.dok_array,
    ss.bsr_matrix, ss.bsr_array
]



