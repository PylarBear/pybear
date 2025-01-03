# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import warnings
import numpy as np
import pandas as pd

from pybear.utilities import nan_mask



def cast_to_ndarray(X):

    """
    Convert the container of OBJECT to numpy.ndarray.


    Parameters
    ----------
    OBJECT:
        array-like of shape (n_samples, n_features) or (n_samples,) -
        The array-like data to be converted to NDArray.


    Return
    ------
    -
        OBJECT: the original data converted to NDArray.

    """

    # block unmentionables -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    try:
        # ss dok is not passing iter()
        # make a short circuit for all scipy sparse
        if hasattr(X, 'toarray'):
            raise UnicodeError
        iter(X)
        if isinstance(X, (str, dict)):
            raise Exception
    except UnicodeError:
        # skip out for scipy sparse
        pass
    except:
        raise TypeError(
            f"cast_to_ndarray: X must be an iterable that can be converted to "
            f"a numpy ndarray."
        )


    _suffix = (
        f"\nPass X as a numpy ndarray, pandas dataframe, pandas series, "
        f"dask array, dask dataframe, or dask series."
    )
    if isinstance(X, np.recarray):
        raise TypeError(
            f"cast_to_ndarray: OBJECT is a numpy recarray. " + _suffix
        )

    if isinstance(X, np.ma.core.MaskedArray):
        raise TypeError(
            f"cast_to_ndarray: OBJECT is a numpy masked array. " + _suffix
        )
    # END block unmentionables -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    # IF ss CONVERT TO np
    # do this before nan_mask, otherwise would need to do fancy mask things
    # on ss.data attribute separate from all other formats.
    try:
        X = X.toarray()
    except:
        pass


    # pandas may have funky nan-likes not recognized by numpy. not only do
    # these prevent pd objects from going to numpy nicely via to_numpy(),
    # they also cause ValueError on dask ddf when trying to do compute().
    # forget all those headaches and standardize all nan-likes to
    # numpy.nan with warning.
    _nan_mask = nan_mask(X)
    if np.sum(_nan_mask):
        warnings.warn(
            'The passed dataframe/series object has nan-like values.'
            '\nReplacing all nan-like values with numpy.nan.'
        )
        X[_nan_mask] = np.nan
    del _nan_mask



    # IF dask CONVERT TO np/pd
    try:
        X = X.compute()
    except:
        pass


    # IF pd CONVERT TO np
    if isinstance(X, (pd.core.series.Series, pd.core.frame.DataFrame)):
        X = X.to_numpy()



    X = np.array(X)

    # *** X MUST BE np ***

    if not isinstance(X, np.ndarray):
        raise TypeError(f"X is an invalid data-type {type(X)}")

    return X









