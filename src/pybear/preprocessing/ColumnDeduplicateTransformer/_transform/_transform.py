# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from .._type_aliases import DataContainer
import numpy.typing as npt

import numpy as np
import pandas as pd






def _transform(
    X: DataContainer,
    _column_mask: npt.NDArray[bool]
) -> DataContainer:

    """
    Remove the duplicate columns from X as indicated in the _column_mask
    vector.


    Parameters
    ----------
    X:
        {array-like, scipy sparse matrix} of shape (n_samples,
        n_features) - The data to be deduplicated.
    _column_mask:
        list[bool] - A boolean vector of shape (n_features,) that
        indicates which columns to keep (True) and which columns to
        delete (False).


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
    elif hasattr(X, 'toarray'):     # scipy sparse
        _dtype = X.dtype
        _format = type(X)

        return _format(X.tocsc()[:, _column_mask]).astype(_dtype)
    else:
        raise Exception









