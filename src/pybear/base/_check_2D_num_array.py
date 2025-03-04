# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Iterable, Optional
import numpy.typing as npt
from typing_extensions import TypeAlias, Union

import numbers

import pandas as pd
import polars as pl
import scipy.sparse as ss
import dask.array as da
import dask.dataframe as ddf

from ._check_1D_num_sequence import check_1D_num_sequence


PythonTypes: TypeAlias = Union[list[list], tuple[tuple]]
NumpyTypes: TypeAlias = npt.NDArray
PandasTypes: TypeAlias = pd.DataFrame
PolarsTypes: TypeAlias = pl.DataFrame
SparseTypes: TypeAlias = Union[
    ss.csc_matrix, ss.csc_array, ss.csr_matrix, ss.csr_array,
    ss.coo_matrix, ss.coo_array, ss.dia_matrix, ss.dia_array,
    ss.lil_matrix, ss.lil_array, ss.dok_matrix, ss.dok_array,
    ss.bsr_matrix, ss.bsr_array
]
DaskTypes: TypeAlias = Union[da.Array, ddf.DataFrame]

XContainer: TypeAlias = \
    Union[PythonTypes, NumpyTypes, PandasTypes, PolarsTypes, SparseTypes]



def check_2D_num_array(
    X:XContainer[numbers.Number],
    require_all_finite:Optional[bool] = False
) -> None:

    """
    Validate things that are expected to be 2D arrays of numbers.
    Accepts 2D python built-ins, numpy arrays, pandas dataframes, polars
    dataframes, and all scipy sparse matrices/arrays. Python built-ins
    can be ragged. When 'require_all_finite' is True, every element in
    the array must be an instance of numbers.Number; if there is a
    nan-like or infinity-like value a ValueError will be raised. If
    'require_all_finite' is False, non-finite values are ignored and
    only the finite values must be an instance of numbers.Number. If all
    checks pass then None is returned.


    Parameters
    ----------
    X:
        XContainer[numbers.Number] - something that is expected to be a
        2D array of numbers.
    require_all_finite:
        Optional[bool], default=False - if True, disallow all non-finite
        values, such as nan-like or infinity-like values.


    Raises
    ------
    TypeError:
        for invalid container
    ValueError:
        for non-finite values when 'require_all_finite' is True


    Return
    ------
    -
        None


    Notes
    -----
    Type Aliases

    PythonTypes:
        Union[list[list], tuple[tuple]]

    NumpyTypes:
        npt.NDArray

    PandasTypes:
        pd.DataFrame

    PolarsTypes:
        pl.DataFrame

    SparseTypes:
        Union[
            ss.csc_matrix, ss.csc_array, ss.csr_matrix, ss.csr_array,
            ss.coo_matrix, ss.coo_array, ss.dia_matrix, ss.dia_array,
            ss.lil_matrix, ss.lil_array, ss.dok_matrix, ss.dok_array,
            ss.bsr_matrix, ss.bsr_array
        ]

    XContainer:
        Union[
            PythonTypes, NumpyTypes, PandasTypes, PolarsTypes, SparseTypes
        ]


    Examples
    --------
    >>> from pybear.base import check_2D_num_array
    >>> import numpy as np
    >>> X = np.random.randint(0, 10, (37, 13)).astype(np.float64)
    >>> X[0][8] = np.nan
    >>> X[31][3] = np.inf
    >>> check_2D_num_array(X, require_all_finite=False)

    >>> try:
    ...     check_2D_num_array(X, require_all_finite=True)
    ... except ValueError as e:
    ...     print(e)
    got non-finite values when not allowed

    """


    _err_msg = f"Expected a 2D array of number-like values. "
    _addon = (
        f"\nAccepted containers are 2D python lists, tuples, and sets, "
        f"\nnumpy 2D arrays, pandas dataframes, polars dataframes, and all "
        f"\nscipy sparse matrices/arrays."
    )


    # block disallowed containers -- -- -- -- -- -- -- -- -- -- -- -- --
    if isinstance(X, (pd.Series, pl.Series, ddf.Series)):
        raise TypeError(_err_msg + _addon)

    try:
        # dok gets a free pass
        if isinstance(X, (ss.dok_matrix, ss.dok_array)):
            raise UnicodeError
        # must be iterable
        iter(X)
        # cant be string or dict
        if isinstance(X, (str, dict)):
            raise Exception
        # handle dask or anything with shape attr directly
        if hasattr(X, 'shape'):
            if len(getattr(X, 'shape')) != 2:
                raise Exception
            raise UnicodeError
        if any(map(isinstance, X, ((str, dict) for _ in X))):
            raise Exception
        if not all(map(isinstance, X, (Iterable for _ in X))):
            raise Exception
    except UnicodeError:
        pass
    except Exception as e:
        raise TypeError(f"'expected a 2D array of numbers'")
    # END block disallowed containers -- -- -- -- -- -- -- -- -- -- --


    if hasattr(X, 'toarray'):   # ss
        check_1D_num_sequence(X.tocsr().data, require_all_finite)
    elif isinstance(X, pd.DataFrame):   # pandas
        list(map(
            check_1D_num_sequence,
            X.values,
            (require_all_finite for _ in X.values)
        ))
    elif isinstance(X, pl.DataFrame):   # polars
        list(map(
            check_1D_num_sequence,
            X.rows(),
            (require_all_finite for _ in X.rows())
        ))
    else:
        list(map(
            check_1D_num_sequence,
            X,
            (require_all_finite for _ in X)
        ))







