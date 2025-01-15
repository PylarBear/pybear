# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing import Literal

import numpy as np
import pandas as pd

from ..utilities._nan_masking import nan_mask



def check_dtype(
    X,
    allowed: Literal['numeric', 'any']='any'
) -> None:

    """
    Check that the passed data contains a datatype that is allowed. If
    not, raise ValueError.


    Parameters
    ----------
    X:
        array-like of shape (n_samples, n_features) or (n_samples,). The
        data to be checked for allowed datatype.
    allowed:
        Literal['numeric', 'any'], default='any' - the allowed datatype
        for the data.
        If 'numeric', only allow data that can be converted to dtype
        numpy.float64. If the data cannot be converted, raise ValueError.
        If 'any', allow any datatype.


    Return
    ------
    -
        None.

    """

    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    err_msg = f"'allowed' must be literal 'numeric' or 'any'."

    if not isinstance(allowed, str):
        raise TypeError(err_msg)

    allowed = allowed.lower()

    if allowed not in ['numeric', 'any']:
        raise ValueError(err_msg)

    del err_msg

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    err_msg = f"X can only contain numeric datatypes"

    if allowed == 'numeric':

        # --------------
        # block non-numeric
        if isinstance(X, np.ndarray):
            try:
                X.astype(np.float64)
            except:
                raise ValueError(err_msg)

        elif isinstance(X, (pd.core.series.Series, pd.core.frame.DataFrame)):
            # need to circumvent funky pd nan-likes.
            # .astype(np.float64) is raising when having to convert funky
            # pd nan-likes to float, so pd cant be handled with ndarray
            # above. but X[nan_mask(X)] = np.nan is back-talking to the
            # passed X and mutating it. want to do this without mutating
            # X or making a copy, so scan it column by column (still will
            # there be copies, but smaller)

            if isinstance(X, pd.core.series.Series):
                try:
                    # empiricism shows must use nan_mask not nan_mask_numerical.
                    np.float64(X.copy()[np.logical_not(nan_mask(X))])
                except:
                    raise ValueError(err_msg)

            elif isinstance(X, pd.core.frame.DataFrame):
                for c_idx in range(X.shape[1]):
                    try:
                        _column = X.iloc[:, c_idx]
                        # empiricism shows must use nan_mask not nan_mask_numerical.
                        np.float64(_column[np.logical_not(nan_mask(_column))])
                    except:
                        raise ValueError(err_msg)
                del _column

        elif hasattr(X, 'toarray'):
            # scipy sparse can only be numeric dtype, so automatically good
            pass

        else:
            raise TypeError(
                f":function: 'check_dtype' cannot currently process this "
                f"type of data (only np.ndarray, pd.DataFrame, and scipy). "
                f"got {type(X)}."
            )

    elif allowed == 'any':
        pass



    return





