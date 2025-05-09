# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy as np



def _val_y(_y) -> None:

    """
    Validate y.

    v v v v pizza keep ur finger on this v v v v
    y must be numeric.

    y must be a single label and binary in 0,1.


    Parameters
    ----------
    _y:
        vector-like of shape (n_samples, 1) or (n_samples,) - The target
        for the data.


    Return
    ------
    -
        None

    """


    # y ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    try:
        y_shape = _y.shape
    except:
        try:
            y_shape = np.array(list(_y)).shape
        except:
            raise TypeError(f"'y' must have a 'shape' attribute pizza fix this")

    # pizza make be any 2 uniques
    _err_msg = (
        f"GSTCV can only perform thresholding on vector-like binary targets "
        f"with values in [0,1]. \nPass 'y' as a vector of 0's and 1's."
    )

    if len(y_shape) == 1:
        pass
    elif len(y_shape) == 2:
        if y_shape[1] != 1:
            raise ValueError(_err_msg)
    else:
        raise ValueError(_err_msg)

    # pizza make be any 2 uniques
    if not set(np.unique(_y)).issubset({0, 1}):
        raise ValueError(_err_msg)




