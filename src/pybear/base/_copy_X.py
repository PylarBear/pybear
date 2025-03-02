# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import TypeAlias, Union
import numpy.typing as npt

import numbers
from copy import deepcopy

import numpy as np
import pandas as pd
import scipy.sparse as ss
import polars as pl
import dask.array as da
import dask.dataframe as ddf



PythonTypes: TypeAlias = Union[list, tuple, set]
NumpyTypes: TypeAlias = \
    Union[npt.NDArray[Union[numbers.Number, str]], np.ma.MaskedArray]
PandasTypes: TypeAlias = Union[pd.Series, pd.DataFrame]
PolarsTypes: TypeAlias = Union[pl.Series, pl.DataFrame]
DaskTypes: TypeAlias = Union[da.Array, ddf.Series, ddf.DataFrame]
SparseTypes: TypeAlias = Union[
    ss.csc_matrix, ss.csc_array, ss.csr_matrix, ss.csr_array,
    ss.coo_matrix, ss.coo_array, ss.dia_matrix, ss.dia_array,
    ss.lil_matrix, ss.lil_array, ss.dok_matrix, ss.dok_array,
    ss.bsr_matrix, ss.bsr_array
]

XContainer: TypeAlias = \
    Union[
        PythonTypes, NumpyTypes, PandasTypes,
        PolarsTypes, DaskTypes, SparseTypes
    ]



def copy_X(
    X: XContainer
) -> XContainer:

    """
    Make a deep copy of X. Can take python lists, tuples, and sets,
    numpy arrays and masked arrays, pandas dataframes and series,
    polars dataframes and series, dask arrays, dataframes, and series,
    and scipy sparse matrices/arrays.


    Parameters
    ----------
    X:
        XContainer - the data to be copied.


    Return
    ------
    X:
        XContainer - a deep copy of X.

    """


    # dont use type aliases as long as supporting py39
    if not isinstance(
        X,
            (list, tuple, set, np.ndarray, np.ma.MaskedArray, pd.Series,
             pd.DataFrame, pl.Series, pl.DataFrame, da.Array, ddf.Series,
             ddf.DataFrame, ss.csc_matrix, ss.csc_array, ss.csr_matrix,
             ss.csr_array, ss.coo_matrix, ss.coo_array, ss.dia_matrix,
             ss.dia_array, ss.lil_matrix, ss.lil_array, ss.dok_matrix,
             ss.dok_array, ss.bsr_matrix, ss.bsr_array)
    ):
        raise TypeError(f"copy_X(): unsupported container {type(X)}")


    if hasattr(X, 'clone'):
        _X = X.clone()
    elif isinstance(X, (list, tuple, set)) or not hasattr(X, 'copy'):
        _X = deepcopy(X)
    else:  # has copy() method
        _X = X.copy()


    return _X





