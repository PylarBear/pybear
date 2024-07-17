# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as ddf

from model_selection.GSTCV._type_aliases import (
    XInputType,
    YInputType,
    XSKWIPType,
    YSKWIPType,
    FeatureNamesInType,
    SchedulerType
)






def _handle_X_y_dask(
    X: XInputType,
    y: YInputType = None,
    ) -> tuple[XSKWIPType, YSKWIPType, FeatureNamesInType, int]:

    """



    Process given X and y into dask.array.core.Arrays. All DASK GSTCV internals
    require dask.array.core.Arrays. Accepts objects that can be converted to
    dask arrays, including numpy arrays, pandas dataframes and series,
    and dask dataframes and series. y, if 2 dimensional, is converted to
    a 1 dimensional dask.array.core.Array vector. Pizza, circle around to this
    once you finalize how and where binary-ness of y (cannot take
    multiclass) is to be validated.

    Parameters
    ----------
    X:
        Iterable[Iterable[Union[int, float]]] - The data to be used for
            hyperparameter search.
    y:
        Union[Iterable[int], None] - The target to be used for hyper-
        parameter search.

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
            such as a dask/ pandas dataframe or series, is passed, the
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

    # X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    if isinstance(_X, np.ndarray):
        _X = da.from_array(_X, chunks=_X.shape)

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

    elif isinstance(_X, (ddf.core.Series, ddf.core.DataFrame)):

        try:
            _X = _X.to_frame()
        except:
            pass

        _feature_names_in = np.array(_X.columns)

        _X = _X.to_dask_array(lengths=True)

    else:
        raise TypeError(err_msg('X', _X))

    if len(_X.shape) == 1:
        _X = _X.reshape((len(_X), 1))

    _n_features_in = _X.shape[1]

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

    elif isinstance(_y, (ddf.core.Series, ddf.core.DataFrame)):

        try:
            _y = _y.to_frame()
        except:
            pass

        _y = _y.to_dask_array(lengths=True)

    else:
        raise TypeError(err_msg('y', _y))


    if _y is not None and len(_y.shape) == 2:
        if _y.shape[1] == 1:
            _y = _y.ravel()
        else:
            raise ValueError(f"A multi-column array was passed for y")


    if _y is not None and _X.shape[0] != _y.shape[0]:
        raise ValueError(f"X rows ({_X.shape[0]}) and y rows ({_y.shape[0]}) are not equal")


    return _X, _y, _feature_names_in, _n_features_in













































