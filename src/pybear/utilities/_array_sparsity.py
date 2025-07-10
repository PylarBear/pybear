# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause



import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as ss



def array_sparsity(a) -> float:
    """Calculate the sparsity (percentage of zeros) of an array-like.

    Parameters
    -----------
    a : array_like of shape (n_samples, n_features) or (n_samples,)
        object for which to calculate sparsity. Must not be empty.

    Returns
    -------
    sparsity : float
        percentage of zeros in a.

    Examples
    --------
    >>> import numpy as np
    >>> from pybear.utilities import array_sparsity
    >>> a = np.array([[0,1,0,2,0],[1,0,2,0,3]])
    >>> array_sparsity(a)
    50.0

    """

    # if is known container skip validation ** * ** * ** * ** * ** * **
    _skip_validation = 0
    _skip_validation += hasattr(a, 'ravel')  # np
    _skip_validation += hasattr(a, 'to_numpy')   # pd, pl
    _skip_validation += hasattr(a, 'toarray')    # ss
    # END if is known container skip validation ** * ** * ** * ** * ** *


    err_msg = (f"'a' must be a non-empty array-like that can be "
               f"converted to numpy.ndarray.")

    if hasattr(a, 'size') and a.size == 0:
        raise ValueError(err_msg)

    if not _skip_validation:

        try:
            list(iter(a))
            if isinstance(a, (str, dict)):
                raise Exception
        except Exception as e:
            raise TypeError(err_msg)


    if isinstance(a, np.ndarray):
        return float((a == 0).astype(np.int32).sum() / a.size * 100)
    elif isinstance(a, (pd.core.series.Series, pd.core.frame.DataFrame)):
        return float((a == 0).astype(np.int32).sum() / a.size * 100)
    elif isinstance(a, (pl.series.Series, pl.dataframe.DataFrame)):
        return float((a == 0).cast(pl.Int32).sum() / a.size * 100)
    elif hasattr(a, 'toarray'):
        return
    else:
        try:
            a = np.array(list(map(list, a)))
        except:
            try:
                a = np.array(list(a))
            except:
                raise TypeError(f"failed to convert 'a' to a numpy array")

        if a.size == 0:
            raise ValueError(err_msg)

        return float((a == 0).astype(np.int32).sum() / a.size * 100)




