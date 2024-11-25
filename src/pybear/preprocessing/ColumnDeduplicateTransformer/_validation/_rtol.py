# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import numpy as np
from numbers import Real



def _val_rtol(_rtol: Real) -> None:

    """
    Verify rtol is a non-boolean, real, non-negative number that is
    accepted by numpy allclose.


    Parameters
    ----------
    _rtol:
        Real - the relative difference tolerance for equality. Must be a
            non-boolean, non-negative, real number. See numpy.allclose.


    Return
    ------
    -
        None


    """


    err_msg = (f"'rtol' must be a non-boolean, real, non-negative number "
               f"that is accepted by numpy allclose")


    if not isinstance(_rtol, Real):
        raise ValueError(err_msg)

    if isinstance(_rtol, bool):
        raise ValueError(err_msg)

    if _rtol < 0:
        raise ValueError(err_msg)

    X1 = np.random.uniform(0, 1, 20)

    np.allclose(X1, X1, rtol=_rtol, atol=1e-6)






