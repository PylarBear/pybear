# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




import numpy as np
import pandas as pd





def cast_to_ndarray(X):

    """
    Convert the container of OBJECT to numpy.ndarray.

    Does not accept python built-in containers (list, set, tuple).
    pybear strongly encourages (even requires) you to pass your data
    in third party containers such as numpy arrays.

    Does not do any nan handling. This module uses methods that are
    native to the containers to convert them to numpy ndarray. pybear
    cannot handle every possible edge case when converting data to
    numpy ndarray. Let the native handling allow or disallow nan-likes.
    This forces the user to clean up their own data outside of pybear.


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

    # block unsupported containers -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    try:
        # ss dok is not passing iter()
        # make a short circuit for all scipy sparse
        if hasattr(X, 'toarray'):
            raise UnicodeError
        iter(X)
        if isinstance(X, (str, dict)):
            raise Exception
        if isinstance(X, (set, tuple, list)):
            raise MemoryError
    except UnicodeError:
        # skip out for scipy sparse
        pass
    except MemoryError:
        raise TypeError(
            f"cast_to_ndarray does not currently accept python built-in "
            f"iterables (set, list, tuple). Pass X as a container that "
            f"has a 'shape' attribute."
        )
    except:
        raise TypeError(
            f"cast_to_ndarray: X must be a vector-like or array-like that "
            f"can be converted to a numpy ndarray."
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
    # END block unsupported containers -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    # IF ss CONVERT TO np
    # do this before nan_mask, otherwise would need to do fancy mask things
    # on ss.data attribute separate from all other formats.
    try:
        X = X.toarray()
    except:
        pass


    # IF dask CONVERT TO np/pd
    if hasattr(X, 'compute'):
        X = X.compute()


    # IF pd CONVERT TO np
    if isinstance(X, (pd.core.series.Series, pd.core.frame.DataFrame)):
        X = X.to_numpy()


    # # IF polars CONVERT TO np
    # if isinstance(X, (pl.DataFrame)):
    #     X = X.to_numpy()


    X = np.array(X)

    # *** X MUST BE np ***

    if not isinstance(X, np.ndarray):
        raise TypeError(f"X is an invalid data-type {type(X)}")

    return X









