# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from .._type_aliases import DataType
import numpy.typing as npt

import numpy as np
import pandas as pd

from ....utilities._nan_masking import nan_mask



def _column_getter(
    _X: DataType,
    _col_idx: int,
) -> npt.NDArray[any]:

    """
    This supports _find_duplicates. Handles the mechanics of extracting
    a column from the various data container types. Return extracted
    column as a numpy vector. In the case of scipy sparse, the columns
    are not converted to dense, but the dense indices in the 'indices'
    attribute and the values in the 'data' attribute are hstacked
    together and that is sent for equality test.


    Parameters
    ----------
    _X:
        DataType - The data to be deduplicated.
    _col_idx:
        int - the column index of the column to be extracted from _X.


    Return
    ------
    -
        column: NDArray[any] - The column corresponding to the given
            index.


    """


    assert isinstance(_col_idx, int)

    if isinstance(_X, np.ndarray):
        column = _X[:, _col_idx]
    elif isinstance(_X, pd.core.frame.DataFrame):
        column = _X.iloc[:, _col_idx].to_numpy()
    elif hasattr(_X, 'toarray'):    # scipy sparse
        # instead of expanding the column to dense np, get the indices
        # and values out of sparse column using the 'indices' and 'data'
        # attributes, hstack them, and send that off for equality test

        # Extract the data and indices of the column
        c1 = _X.getcol(_col_idx).tocsc()  # tocsc() is important, must stay
        column = np.hstack((c1.indices, c1.data))
        del c1

        # old code that converts a ss column to np array
        # _X_wip = _X.copy().tocsc()
        # column = _X_wip[:, [_col_idx]].toarray().ravel()
        # del _X_wip
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
        column[nan_mask(column)] = np.nan
    except:
        pass

    return column








