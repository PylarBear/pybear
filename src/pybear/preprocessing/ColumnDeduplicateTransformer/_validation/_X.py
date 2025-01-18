# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import DataContainer

import numpy as np
import pandas as pd




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


    if not isinstance(_X, (np.ndarray, pd.core.frame.DataFrame)) and not \
        hasattr(_X, 'toarray'):
        raise TypeError(
            f"invalid container for X: {type(_X)}. X must be numpy array, "
            f"pandas dataframe, or any scipy sparce matrix / array."
        )










