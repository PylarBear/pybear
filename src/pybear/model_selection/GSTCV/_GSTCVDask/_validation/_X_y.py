# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import itertools

from dask import compute
import dask.array as da
import dask.dataframe as ddf



def _val_X_y(
    _X,
    _y
) -> None:

    """
    Validate X and y.

    v v v v pizza keep ur finger on this v v v v
    All GSTCVDask internals require dask.array.core.Arrays. Accepts
    objects that can be converted to dask arrays, including numpy arrays,
    pandas dataframes and series, and dask dataframes and series.

    v v v v pizza keep ur finger on this v v v v
    Both X and y must be numeric (i.e., can pass a test where they are
    converted to np.uint8.) GSTCV (and most estimators) cannot accept
    non-numeric data.

    y must have the same number of samples as X.
    y must be a single label and binary in 0,1.


    Parameters
    ----------
    _X:
        array-like of shape (n_samples, n_features) - The data. Must
        be a dask object.
    _y:
        vector-like of shape (n_samples, 1) or (n_samples,) - The target
        for the data. Must be a dask object.


    Return
    ------
    -
        None

    """


    err_msg = lambda _name, _object: (f"{_name} was passed with unsupported "
        f"data type '{type(_object)}'. \nGSTCVDask only supports dask arrays.")
    #  both dask_ml KFold & LogisticRegression expressly block dask ddf

    # X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    if not isinstance(_X, da.core.Array):
        raise TypeError(err_msg('X', _X))

    X_shape = compute(*_X.shape)

    if len(X_shape) not in [1, 2]:
        raise ValueError(f"'X' must be a 1 or 2 dimensional object")

    # v v v v pizza keep ur finger on this v v v v
    # try to convert first block of _X to np.uint8, to prove only numerical
    # try:
    #     _X.blocks[0].astype(np.uint8).compute()
    # except:
    #     raise ValueError(f"dtype='numeric' is not compatible with arrays "
    #         f"of bytes/strings. Convert your data to numeric values "
    #         f"explicitly instead."
    #     )

    # v v v v pizza keep ur finger on this v v v v
    # under certain circumstances (e.g. using ddf.to_dask_array() without
    # specifying 'lengths') array chunk sizes can become np.nan along the
    # row dimension. This causes an error in ravel(). Ensure chunks are
    # specified.
    # if isinstance(_X, da.core.Array):
    #     X_block_dims = list(compute(*itertools.chain(*map(da.shape, _X.blocks))))
    #     try:
    #         list(map(int, X_block_dims))
    #     except:
    #         raise ValueError(f"X chunks are not defined. rechunk X.")
    #     del X_block_dims

    # END X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # y ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    if not isinstance(_y, da.core.Array):
        raise TypeError(err_msg('y', _y))

    y_shape = compute(*_y.shape)

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

    if isinstance(_y, da.core.Array):
        _unique = set(da.unique(_y).compute())
    elif isinstance(_y, (ddf.DataFrame, ddf.Series)):
        _unique = set(da.unique(_y.to_dask_array(lengths=True)).compute())

    if not _unique.issubset({0, 1}):

        raise ValueError(
            f"GSTCVDask can only perform thresholding on binary targets "
            f"with values in [0,1]. \nPass 'y' as a vector of 0's and "
            f"1's."
        )

    del _unique


