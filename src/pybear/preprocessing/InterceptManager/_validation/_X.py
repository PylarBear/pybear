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
    Validate the data. Cannot be None, must be numpy ndarray, pandas
    dataframe, or scipy sparse matrix/array.

    All other validation of the data is handled in the individual class
    methods by pybear.base.validate_data.


    Parameters
    ----------
    _X:
        {array-like, scipy sparse matrix} of shape (n_samples,
        n_features) - the data.


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










