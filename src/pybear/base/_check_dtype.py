# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing import Literal

import numpy as np



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

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    if allowed == 'numeric':

        if not hasattr(X, 'astype'):
            raise ValueError(
                f"check_dtype: 'X' must have an 'astype' attribute."
            )

        try:
            X.astype(np.float64)
        except:
            raise ValueError(
                f"'X' must be numeric but could not be converted to "
                f"numpy.float64 datatype."
            )

    elif allowed == 'any':
        pass



    return





