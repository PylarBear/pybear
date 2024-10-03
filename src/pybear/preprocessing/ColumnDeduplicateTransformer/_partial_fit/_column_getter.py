# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import DataType
import numpy.typing as npt

import numpy as np
import pandas as pd

import scipy.sparse as ss


def _column_getter(
    _X: DataType,
    _col_idx1: int,
    _col_idx2: int
) -> tuple[npt.NDArray[any], npt.NDArray[any]]:

    """
    This supports _find_duplicates. Handles the mechanics of getting
    columns for the various input types.

    Parameters
    ----------
    _X:
        DataType - The data to be deduplicated.
    _col_idx1:
        int - the first column in the comparison pair.
    _col_idx2:
        int - the second column in the comparison pair.


    Return
    ------
    -
        column1, column2: tuple[NDArray[any], NDArray[any]] - The columns
        corresponding to the given indices.


    """

    if isinstance(_X, np.ndarray):
        column1 = _X[:, _col_idx1]
        column2 = _X[:, _col_idx2]
    elif isinstance(_X, pd.core.frame.DataFrame):
        column1 = _X.iloc[:, _col_idx1]
        column2 = _X.iloc[:, _col_idx2]
    elif isinstance(_X, (ss._csr.csr_matrix, ss._csc.csc_matrix,
        ss._coo.coo_matrix, ss._dia.dia_matrix, ss._lil.lil_matrix,
        ss._dok.dok_matrix, ss._bsr.bsr_matrix, ss._csr.csr_array,
        ss._csc.csc_array, ss._coo.coo_array, ss._dia.dia_array,
        ss._lil.lil_array, ss._dok.dok_array, ss._bsr.bsr_array)):

        _X_wip = _X.copy().tocsc()
        column1 = _X_wip[:, [_col_idx1]].toarray().ravel()
        column2 = _X_wip[:, [_col_idx2]].toarray().ravel()
    else:
        raise TypeError(f"invalid data type '{type(_X)}'")


    return column1, column2








