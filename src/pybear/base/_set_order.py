# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Literal
import numpy.typing as npt

import numpy as np




def set_order(
    X: npt.NDArray,
    *,
    order: Literal['C', 'F']="C"
) -> npt.NDArray:

    """
    Set the memory layout of X. X must be a numpy ndarray. 'C' is
    row-major order, 'F' is column major order.

    For 1D and trivial 2D (shape=(10, 1) or (1, 10)) numpy arrays, the
    'flags' attribute will report both C_CONTIGUOUS and F_CONTIGUOUS as
    True. This is because these arrays are a single continuous block of
    memory with no dimensions to reorder. Both C_CONTIGUOUS and
    F_CONTIGUOUS are True because there is no ambiguity in accessing
    elements — the memory layout trivially satisfies both definitions.


    Parameters
    ----------
    X:
        NDArray - the numpy array for which to set the memory layout.
    order:
        Literal['c', 'C', 'f', 'F'] - the memory layout for X. 'C' is
        row-major order, 'F' is column-major order.


    Return
    ------
    -
        X: NDArray - X in the desired memory layout.

    """


    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    if not isinstance(X, np.ndarray):
        raise TypeError(f"'X' must be a numpy ndarray.")

    err_msg = f"'order' must be a string literal 'C' or 'F', not case sensitive."
    if not isinstance(order, str):
        raise TypeError(err_msg)

    order = order.upper()

    if order not in ['C', 'F']:
        raise ValueError(err_msg)

    del err_msg
    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    if order == 'C':
        X = np.ascontiguousarray(X)
    elif order == 'F':
        X = np.asfortranarray(X)

    return X




