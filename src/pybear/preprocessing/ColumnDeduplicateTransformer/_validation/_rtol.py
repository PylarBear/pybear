# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import numpy as np



def _val_rtol(_rtol: float) -> None:

    """
    Verify rtol is a non-boolean number that is accepted by numpy
    allclose.


    Parameters
    ----------
    _rtol:
        float - the relative difference tolerance for equality.


    Return
    ------
    -
        None


    """

    if isinstance(_rtol, bool):
        raise TypeError(f"'rtol' cannot be bool")

    X1 = np.random.uniform(0, 1, 20)

    np.allclose(X1, X1, rtol=_rtol, atol=1e-6)




