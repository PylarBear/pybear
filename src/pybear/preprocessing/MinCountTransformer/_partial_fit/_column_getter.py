# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import InternalXContainer, DataType
import numpy.typing as npt

import numpy as np
import pandas as pd
import scipy.sparse as ss

from ....utilities._nan_masking import nan_mask



def _column_getter(
    _X: InternalXContainer,
    _col_idx: int,
) -> npt.NDArray[DataType]:

    """
    Handles the mechanics of extracting a column from the various data
    container types. Returns extracted column as a numpy vector. In the
    case of scipy sparse, the columns are not converted to dense, only
    the values in the 'data' attribute are returned.


    Parameters
    ----------
    _X:
        InternalXContainer - The data to undergo minimum frequency
        thresholding. The data container must be indexable. Therefore,
        scipy sparse coo, dia, and bsr matrix/arrays are not permitted
        in this module. There is no conditioning of the data here and
        this module expects to receive it in suitable form.
    _col_idx:
        int - the column index of the column to be extracted from _X.


    Return
    ------
    -
        column: NDArray[any] - The column corresponding to the given
            index. If extracted from a scipy sparse matrix/array, it is
            the 'data' attribute of that single column.


    """

    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    assert isinstance(_col_idx, int)
    assert _col_idx in range(-_X.shape[1], _X.shape[1])

    assert (isinstance(_X, (np.ndarray, pd.core.frame.DataFrame))
            or hasattr(_X, 'toarray'))

    assert not isinstance(_X,
        (ss.coo_matrix, ss.coo_array, ss.dia_matrix,
         ss.dia_array, ss.bsr_matrix, ss.bsr_array)
    )
    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    if isinstance(_X, np.ndarray):
        column = _X[:, _col_idx]
    elif isinstance(_X, pd.core.frame.DataFrame):
        column = _X.iloc[:, _col_idx].to_numpy()
    elif hasattr(_X, 'toarray'):    # scipy sparse
        # instead of expanding the column to dense np, get the data
        # values out of the sparse column using the 'data' attribute.
        # the difference _X.shape[0] - len(ss.data) is the number of
        # zeros in the column.
        column = _X[:, [_col_idx]].tocsc().data
        # .tocsc() is important, dok, at least, doesnt have a .data attr
    else:
        raise TypeError(f"invalid data type '{type(_X)}'")

    # this assignment must stay here.
    # np.nan assignment excepts on dtype int array, so ask for forgiveness
    try:
        column[nan_mask(column)] = np.nan
    except:
        pass

    return column








