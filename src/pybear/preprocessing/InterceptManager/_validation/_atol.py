# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import numpy as np



def _val_atol(_atol: float) -> None:

    """
    Verify atol is a non-boolean number that is accepted by numpy
    allclose.


    Parameters
    ----------
    _atol:
        float - the absolute difference tolerance for equality.


    Return
    ------
    -
        None


    """

    if isinstance(_atol, bool):
        raise TypeError(f"'atol' cannot be bool")

    X1 = np.random.uniform(0, 1, 20)

    try:
        np.allclose(X1, X1, rtol=1e-6, atol=_atol)
    except Exception as e:
        raise e






