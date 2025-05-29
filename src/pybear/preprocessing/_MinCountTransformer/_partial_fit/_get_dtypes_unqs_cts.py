# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
from .._type_aliases import (
    InternalXContainer,
    DataType
)

import itertools
import numbers

import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as ss
import joblib

from .._partial_fit._columns_getter import _columns_getter
from .._partial_fit._parallel_dtypes_unqs_cts import _parallel_dtypes_unqs_cts



def _get_dtypes_unqs_cts(
    _X: InternalXContainer,
    _n_jobs: Union[numbers.Integral, None]
) -> tuple[str, dict[DataType, int]]:

    """
    Parallelized collection of dtypes, uniques, and counts for every
    column in X.


    Parameters
    ----------
    _X:
        InternalXContainer - The pizza.
    _n_jobs:
        Union[numbers.Integral, None] - the number of parallel jobs
        to use when scanning X. -1 means using all pizzas.


    Returns
    -------
    -
        list[tuple[str, dict[DataType, int]]] - a list of tuples, one
        tuple for each column in X. Each tuple holds the pybear-assigned
        dtype for the column and a dictionary with the uniques in the
        column as keys and their respective frequencies as values, and pizza.

    """


    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    assert isinstance(_X,
        (np.ndarray, pd.core.frame.DataFrame, pl.DataFrame, ss.csc_array,
         ss.csc_matrix)
    )

    assert isinstance(_n_jobs, (int, type(None)))
    if _n_jobs:
        assert _n_jobs >= -1
    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # out is list[tuple[str, dict[DataType, int]]]
    # the idxs of the list match the idxs of the data

    # number of columns to send in a job to joblib
    _n_cols = 1000



    if _X.shape[1] < 2 * _n_cols:
        DTYPE_UNQS_CTS_TUPLES = _parallel_dtypes_unqs_cts(
            _columns_getter(_X, tuple(range(_X.shape[1])))
        )
    else:
        # DONT HARD-CODE backend, ALLOW A CONTEXT MANAGER TO SET
        with joblib.parallel_config(prefer='processes', n_jobs=_n_jobs):
            DTYPE_UNQS_CTS_TUPLES = joblib.Parallel(return_as='list')(
                    joblib.delayed(_parallel_dtypes_unqs_cts)(
                        _columns_getter(
                            _X,
                            tuple(range(i, min(i + _n_cols, _X.shape[1])))
                        )
                    ) for i in range(0, _X.shape[1], _n_cols)
                )

        DTYPE_UNQS_CTS_TUPLES = list(itertools.chain(*DTYPE_UNQS_CTS_TUPLES))


    return DTYPE_UNQS_CTS_TUPLES




