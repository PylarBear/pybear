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




def _transform(
    X: DataType,
    _column_mask: npt.NDArray[bool]
) -> DataType:

    """
    Remove the duplicate columns from X as indicated in the _column_mask
    vector.


    Parameters
    ----------
    X:
        {array-like, scipy sparse matrix} of shape (n_samples,
        n_features) - The data to be deduplicated.
    _column_mask:
        NDArray[bool] - A boolean vector that indicates which
        columns to keep (True) and which columns to delete (False).


    Return
    ------
    -
        X: {array-like, scipy sparse matrix} of shape (n_samples,
            n_features - n_features_removed) - The deduplicated data.

    """


    if isinstance(X, np.ndarray):
        return X[:, _column_mask]
    elif isinstance(X, pd.core.frame.DataFrame):
        return X.loc[:, _column_mask]
    elif isinstance(X, (ss._csr.csr_matrix, ss._csc.csc_matrix,
        ss._coo.coo_matrix, ss._dia.dia_matrix, ss._lil.lil_matrix,
        ss._dok.dok_matrix, ss._bsr.bsr_matrix, ss._csr.csr_array,
        ss._csc.csc_array, ss._coo.coo_array, ss._dia.dia_array,
        ss._lil.lil_array, ss._dok.dok_array, ss._bsr.bsr_array)):

        _dtype = X.dtype
        _format = type(X)

        return _format(X.tocsc()[:, _column_mask]).astype(_dtype)
    else:
        raise Exception









