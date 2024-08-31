# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import itertools
import numpy as np
import pandas as pd
from dask import compute
import dask.array as da
import dask.dataframe as ddf
import dask_expr._collection as ddf2
from model_selection.GSTCV._type_aliases import (
    XInputType,
    YInputType,
    XSKWIPType,
    YSKWIPType,
    FeatureNamesInType
)



def _handle_X_y_dask(
    X: XInputType,
    y: YInputType = None,
    ) -> tuple[XSKWIPType, YSKWIPType, FeatureNamesInType, int]:

    """
    Process given X and y into dask.array.core.Arrays. All GSTCVDask
    internals require dask.array.core.Arrays. Accepts objects that can
    be converted to dask arrays, including numpy arrays, pandas dataframes
    and series, and dask dataframes and series. y, if 2 dimensional, is
    converted to a 1 dimensional dask.array.core.Array vector.

    Validation always checks for 2 things. First, both X and y must be
    numeric (i.e., can pass a test where they are converted to np.uint8.)
    GSTCV (and most estimators) cannot accept non-numeric data. Secondly,
    y must be a single label and binary in 0,1.


    Parameters
    ----------
    X:
        Iterable[Iterable[Union[int, float]]] - The data to be processed.
    y:
        Union[Iterable[int], None] - The target to be processed.

    Return
    ------
    -
        _X:
            dask.array.core.Array[Union[int, float]] - the given X array
            converted to dask.array.core.Array.
        _y:
            dask.array.core.Array[Union[int, float]] - the given y vector /
            array converted to a dask.array.core.Array vector.
        _feature_names_in:
            Union[NDArray[str], None] - if an object that has column names,
            such as a dask/pandas dataframe or series, is passed, the
            column names are extracted and returned as a numpy vector.
            Otherwise, None is returned.
        _n_features_in:
            int - the number of columns in X.


    """


    err_msg = lambda _name, _object: (f"{_name} was passed with unknown "
        f"data type '{type(_object)}'. Use dask array, dask series, "
        f"dask dataFrame, numpy array, pandas series, pandas dataframe.")

    _feature_names_in = None

    _X = X
    _y = y

    try:
        X_shape = compute(*_X.shape)
    except:
        raise TypeError(err_msg('X', _X))

    # X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    if isinstance(_X, np.ndarray):
        _X = da.from_array(_X, chunks=X_shape)

    elif isinstance(_X, da.core.Array):
        pass

    elif isinstance(_X, (pd.Series, pd.DataFrame)):
        try:
            _X = _X.to_frame()
        except:
            pass

        _feature_names_in = np.array(_X.columns)

        _X = _X.to_numpy()

        _X = da.from_array(_X, chunks=_X.shape)

    elif isinstance(_X,
        (ddf.core.Series, ddf.core.DataFrame, ddf2.Series, ddf2.DataFrame)):

        try:
            _X = _X.to_frame()
        except:
            pass

        _feature_names_in = np.array(_X.columns)

        _X = _X.to_dask_array(lengths=True)

    else:
        raise TypeError(err_msg('X', _X))


    X_block_dims = list(compute(*itertools.chain(*map(da.shape, _X.blocks))))
    try:
        list(map(int, X_block_dims))
    except:
        raise ValueError(f"X chunks are not defined. rechunk X.")
    del X_block_dims


    if len(X_shape) == 1:
        _X = _X.reshape((X_shape[0], 1))
    elif len(X_shape) == 2:
        pass
    else:
        raise ValueError(f"'X' must be a 1 or 2 dimensional object")

    X_shape = compute(*_X.shape)

    _n_features_in = X_shape[1]

    # try to convert first block of _X to np.uint8, to prove only numerical
    try:
        _X.blocks[0].astype(np.uint8).compute()
    except:
        raise ValueError(f"dtype='numeric' is not compatible with arrays "
            f"of bytes/strings. Convert your data to numeric values "
            f"explicitly instead."
        )

    # END X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # y ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    if isinstance(_y, type(None)):
        pass
    elif isinstance(_y, np.ndarray):
        _y = da.from_array(_y, chunks=_y.shape)
    elif isinstance(_y, da.core.Array):
        pass
    elif isinstance(_y, (pd.Series, pd.DataFrame)):
        try:
            _y = _y.to_frame()
        except:
            pass

        _y = _y.to_numpy()
        _y = da.from_array(_y, chunks=_y.shape)

    elif isinstance(_y,
        (ddf.core.Series, ddf.core.DataFrame, ddf2.Series, ddf2.DataFrame)):

        try:
            _y = _y.to_frame()
        except:
            pass

        _y = _y.to_dask_array(lengths=True)

    else:
        raise TypeError(err_msg('y', _y))


    if _y is not None:

        y_shape = compute(*_y.shape)

        # under certain circumstances (e.g. using ddf.to_dask_array() without
        # specifying 'lengths') array chunk sizes can become np.nan along the
        # row dimension. This causes an error in ravel(). Ensure chunks are
        # specified.
        y_block_dims = list(compute(*itertools.chain(*map(da.shape, _y.blocks))))
        try:
            list(map(int, y_block_dims))
        except:
            raise ValueError(f"y chunks are not defined. rechunk y.")
        del y_block_dims


        _, __ = X_shape[0], y_shape[0]
        if _ != __:
            raise ValueError(
                f"Found input variables with inconsistent numbers of samples: "
                f"[{__}, {_}]"
            )
        del _, __

        if len(y_shape) == 1:
            pass
        elif len(y_shape) == 2:
            if y_shape[1] == 1:
                _y = _y.ravel()
                # dont need to get new shape
            else:
                raise ValueError(f"Classification metrics can't handle a mix of "
                    f"multilabel-indicator and binary targets")
        else:
            raise ValueError(f"'y' must be a 1 or 2 dimensional object")

        if not set(da.unique(_y).compute()).issubset({0, 1}):
            raise ValueError(f"GSTCVDask can only perform thresholding on "
                f"binary targets with values in [0,1]. Pass 'y' as a "
                f"vector of 0's and 1's.")


    return _X, _y, _feature_names_in, _n_features_in













































