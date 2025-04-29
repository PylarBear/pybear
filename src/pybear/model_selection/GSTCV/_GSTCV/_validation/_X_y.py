# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy as np



def _val_X_y(_X, _y) -> None:

    """
    Validate X and y.

    v v v v pizza keep ur finger on this v v v v
    Both X and y must be numeric (i.e., can pass a test where they are
    converted to np.uint8.) GSTCV (and most estimators) cannot accept
    non-numeric data.

    y must have the same number of samples as X.
    y must be a single label and binary in 0,1.


    Parameters
    ----------
    _X:
        array-like of shape=(n_samples, n_features) - The data.
    _y:
        vector-like of shape=(n_samples, 1) or (n_samples,) - The target
        for the data.


    Return
    ------
    -
        None

    """


    # X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    try:
        X_shape = _X.shape
    except:
        raise TypeError(f"'X' must have a 'shape' attribute")

    if len(X_shape) not in [1, 2]:
        raise ValueError(f"'X' must be a 1 or 2 dimensional object")


    # pizza keep ur finger on this
    # try to convert _X to np.uint8, to prove only numerical
    # try:
    #     _X.astype(np.uint8)
    # except:
    #     raise ValueError(f"dtype='numeric' is not compatible with arrays "
    #         f"of bytes/strings. Convert your data to numeric values "
    #         f"explicitly instead."
    #     )
    # END X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # y ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    if _y is not None:

        try:
            y_shape = _y.shape
        except:
            raise TypeError(f"'y' must have a 'shape' attribute")


        # v v v v pizza keep ur finger on this v v v v
        # under certain circumstances (e.g. using ddf.to_dask_array() without
        # specifying 'lengths') array chunk sizes can become np.nan along the
        # row dimension. This causes an error in ravel(). Ensure chunks are
        # specified.
        # if isinstance(_y, da.core.Array):
        #     y_block_dims = list(compute(*itertools.chain(*map(da.shape, _y.blocks))))
        #     try:
        #         list(map(int, y_block_dims))
        #     except:
        #         raise ValueError(f"y chunks are not defined. rechunk y.")
        #     del y_block_dims


        if X_shape[0] != y_shape[0]:
            raise ValueError(
                f"Found input variables with inconsistent numbers of samples: "
                f"[{y_shape[0]}, {X_shape[0]}]"
            )

        if len(y_shape) == 1:
            pass
        elif len(y_shape) == 2:
            if y_shape[1] != 1:
                raise ValueError(
                    f"Classification metrics can't handle a mix of "
                    f"multilabel-indicator and binary targets"
                )
        else:
            raise ValueError(f"'y' must be a 1 or 2 dimensional object")

        if not set(np.unique(_y)).issubset({0, 1}):
            raise ValueError(
                f"GSTCV can only perform thresholding on binary targets "
                f"with values in [0,1]. \nPass 'y' as a vector of 0's and "
                f"1's."
            )




