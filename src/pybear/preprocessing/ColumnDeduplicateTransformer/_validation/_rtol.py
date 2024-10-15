# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import numpy as np


def _val_rtol(_rtol: float) -> None:

    if isinstance(_rtol, bool):
        raise TypeError(f"'rtol' cannot be bool")

    X1 = np.random.uniform(0, 1, 20)

    try:
        np.allclose(X1, X1, rtol=_rtol, atol=1e-6)
    except Exception as e:
        raise e




