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

from ....utilities._nan_masking import nan_mask



def _column_getter(
    _X: DataType,
    _col_idx1: int,
    _col_idx2: int
) -> tuple[npt.NDArray[any], npt.NDArray[any]]:

    """
    This supports _find_duplicates. Handles the mechanics of extracting
    columns from the various data container types. Return extracted
    columns as numpy vectors.


    Parameters
    ----------
    _X:
        DataType - The data to be deduplicated.
    _col_idx1:
        int - the first column index in the comparison pair.
    _col_idx2:
        int - the second column index in the comparison pair.


    Return
    ------
    -
        column1, column2: tuple[NDArray[any], NDArray[any]] - The columns
        corresponding to the given indices.


    """

    assert isinstance(_col_idx1, int)
    assert isinstance(_col_idx2, int)

    if isinstance(_X, np.ndarray):
        column1 = _X[:, _col_idx1]
        column2 = _X[:, _col_idx2]
    elif isinstance(_X, pd.core.frame.DataFrame):
        column1 = _X.iloc[:, _col_idx1].to_numpy()
        column2 = _X.iloc[:, _col_idx2].to_numpy()
    elif isinstance(_X, (ss._csr.csr_matrix, ss._csc.csc_matrix,
        ss._coo.coo_matrix, ss._dia.dia_matrix, ss._lil.lil_matrix,
        ss._dok.dok_matrix, ss._bsr.bsr_matrix, ss._csr.csr_array,
        ss._csc.csc_array, ss._coo.coo_array, ss._dia.dia_array,
        ss._lil.lil_array, ss._dok.dok_array, ss._bsr.bsr_array)):

        _X_wip = _X.copy().tocsc()
        column1 = _X_wip[:, [_col_idx1]].toarray().ravel()
        column2 = _X_wip[:, [_col_idx2]].toarray().ravel()
        del _X_wip
    else:
        raise TypeError(f"invalid data type '{type(_X)}'")


    # this assignment must stay here. there was a nan recognition problem
    # that wasnt happening in offline tests of entire data objects
    # holding the gamut of nan-likes but was happening with similar data
    # objects passing thru the CDT machinery. Dont know the reason why,
    # maybe because the columns get parted out, or because they get sent
    # thru the joblib machinery? using nan_mask here and reassigning all
    # nans identified here as np.nan resolved the issue.
    # np.nan assignment excepts on dtype int array, so ask for forgiveness
    try:
        column1[nan_mask(column1)] = np.nan
    except:
        pass

    try:
        column2[nan_mask(column2)] = np.nan
    except:
        pass


    return column1, column2








