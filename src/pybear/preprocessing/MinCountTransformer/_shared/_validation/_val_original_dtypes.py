# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import numpy as np



def _val_original_dtypes(
        _original_dtypes: np.ndarray[str]
    ) -> np.ndarray[str]:

    if not isinstance(_original_dtypes, np.ndarray):
        raise TypeError(f"'_original_dtypes' must be a numpy array")

    ALLOWED = ['bin_int', 'int', 'float', 'obj']

    err_msg = f"entries in '_original_dtypes' must be {', '.join(ALLOWED)}"

    for _dtype in _original_dtypes:
        if not isinstance(_dtype, str):
            raise TypeError(err_msg)

        if _dtype not in ALLOWED:
            raise ValueError(err_msg)

    return _original_dtypes


