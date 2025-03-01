# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import XContainer

import warnings
import numpy as np
import pandas as pd
import scipy.sparse as ss



def _val_X(
    _X: XContainer
) -> None:

    """
    Validate X is a valid data container. Numpy ndarrays, pandas
    dataframes, and all scipy sparse matrices/arrays are allowed.


    Parameters
    ----------
    _X:
        Union[numpy.ndarray, pandas.DataFrame, scipy.sparse] of shape
        (n_samples, n_features). The data to undergo minimum thresholding.


    Return
    ------
    -
        None


    """


    if not isinstance(
        _X,
        (
            np.ndarray,
            pd.core.frame.DataFrame,
            ss._csr.csr_matrix, ss._csc.csc_matrix, ss._coo.coo_matrix,
            ss._dia.dia_matrix, ss._lil.lil_matrix, ss._dok.dok_matrix,
            ss._bsr.bsr_matrix, ss._csr.csr_array, ss._csc.csc_array,
            ss._coo.coo_array, ss._dia.dia_array, ss._lil.lil_array,
            ss._dok.dok_array, ss._bsr.bsr_array
        )
    ):
        raise TypeError(
            f'invalid data container for X, {type(_X)}. must be numpy array, '
            f'pandas dataframe, or any scipy sparse matrix/array.'
        )

    if isinstance(_X, np.rec.recarray):
        raise TypeError(
            f"MCT does not accept numpy recarrays. "
            f"\npass your data as a standard numpy array."
        )

    if np.ma.isMaskedArray(_X):
        warnings.warn(
            f"MCT does not block numpy masked arrays but they are not "
            f"tested. \nuse them at your own risk."
        )














