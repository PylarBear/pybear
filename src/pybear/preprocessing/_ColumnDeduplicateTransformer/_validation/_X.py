# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import DataContainer

import warnings

import numpy as np
import pandas as pd
import polars as pl



def _val_X(
    _X: DataContainer
) -> None:

    """
    Validate the container type of the data. Cannot be None. Otherwise,
    X can be a numpy ndarray, a pandas dataframe, or any other scipy
    sparse matrix / array.

    All other validation of the data is handled by the validate_data
    function at fit, transform, and inverse_transform.


    Parameters
    ----------
    _X:
        {array-like, scipy sparse matrix} of shape (n_samples,
        n_features) - the data to be deduplicated.


    Return
    ------
    -
        None


    """


    if not isinstance(_X, (np.ndarray, pd.core.frame.DataFrame, pl.DataFrame)) and not \
        hasattr(_X, 'toarray'):
        raise TypeError(
            f"invalid container for X: {type(_X)}. X must be numpy array, "
            f"pandas dataframe, or any scipy sparce matrix / array."
        )


    if isinstance(_X, np.rec.recarray):
        raise TypeError(
            f"CDT does not accept numpy recarrays. "
            f"\npass your data as a standard numpy array."
        )

    if np.ma.isMaskedArray(_X):
        warnings.warn(
            f"CDT does not block numpy masked arrays but they are not tested. "
            f"\nuse them at your own risk."
        )







