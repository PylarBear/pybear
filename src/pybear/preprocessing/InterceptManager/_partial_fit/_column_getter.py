# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from .._type_aliases import InternalDataContainer
import numpy.typing as npt

import numpy as np
import pandas as pd
import scipy.sparse as ss

from ....utilities._nan_masking import nan_mask



def _column_getter(
    _X: InternalDataContainer,
    _col_idx: int,
) -> npt.NDArray[any]:

    """
    This supports _find_constants. Handles the mechanics of extracting
    a column from the various data container types. Return extracted
    column as a numpy vector. In the case of scipy sparse, the columns
    are converted to dense.


    Parameters
    ----------
    _X:
        {array-like, scipy sparse} - The data to be searched for constant
        columns. _X must be indexable, which excludes scipy coo, dia, and
        bsr. This module expects _X to be in a valid state when passed,
        and will not condition it.
    _col_idx:
        int - the column index of the column to be extracted from _X.


    Return
    ------
    -
        column: NDArray[any] - The column corresponding to the given
            index.


    """

    # validation ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    assert isinstance(_col_idx, int)
    assert _col_idx in range(-_X.shape[1], _X.shape[1])

    assert not isinstance(_X,
        (ss.coo_matrix, ss.coo_array, ss.dia_matrix,
         ss.dia_array, ss.bsr_matrix, ss.bsr_array)
    )
    # END validation ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    if isinstance(_X, np.ndarray):
        column = _X[:, _col_idx]
    elif isinstance(_X, pd.core.frame.DataFrame):
        column = _X.iloc[:, _col_idx].to_numpy()
    elif hasattr(_X, 'toarray'):    # scipy sparse
        # there are a lot of ifs, ands, and buts if trying to determine
        # if a column is constant just from the dense indices and values.
        # the most elegant way is just to convert to dense, at the expense
        # of some memory swell (but it is only one column, right?)

        # old code that stacks ss column indices and values
        # c1 = _X[:, [_col_idx]]
        # column = np.hstack((c1.indices, c1.data))
        # del c1

        # code that converts a ss column to np array
        column = _X[:, [_col_idx]].toarray().ravel()
    else:
        raise TypeError(f"invalid data type '{type(_X)}'")


    # this assignment must stay here. there was a nan recognition problem
    # that wasnt happening in offline tests of entire data objects
    # holding the gamut of nan-likes but was happening with similar data
    # objects passing thru the IM machinery. Dont know the reason why,
    # maybe because the columns get parted out, or because they get sent
    # thru the joblib machinery? using nan_mask here and reassigning all
    # nans identified here as np.nan resolved the issue.
    # np.nan assignment excepts on dtype int array, so ask for forgiveness
    try:
        column[nan_mask(column)] = np.nan
    except:
        pass

    return column








