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








if __name__ == '__main__':

    import time
    from uuid import uuid4
    _rows, _cols = 100, 30

    X_np_array = np.random.randint(0,10,(_rows, _cols))
    X_COLUMNS = [str(uuid4())[:4] for _ in range(X_np_array.shape[1])]
    X_pd_df = pd.DataFrame(data=X_np_array, columns=X_COLUMNS)
    X_pd_series = pd.Series(X_pd_df.iloc[:, 0], name=X_COLUMNS[0])
    X_dask_array =  da.from_array(X_np_array, chunks=(_rows//5, _cols))
    X_dask_df = ddf.from_pandas(X_pd_df, npartitions=5)
    X_dask_series = X_dask_df.iloc[:, 0]


    y_np_array = np.random.randint(0,10,(_rows, 1))
    y_pd_df = pd.DataFrame(data=y_np_array, columns=['y'])
    y_pd_series = pd.Series(y_pd_df.iloc[:, 0], name='y')
    y_dask_array = da.from_array(y_np_array, chunks=(_rows//5, _cols))
    y_dask_df = ddf.from_pandas(y_pd_df, npartitions=5)
    y_dask_series = y_dask_df.iloc[:, 0]



    for x_idx, X in enumerate((X_np_array, X_pd_df, X_pd_series, X_dask_array, X_dask_df, X_dask_series)):
        for y_idx, y in enumerate((y_np_array, y_pd_df, y_pd_series, y_dask_array, y_dask_df, y_dask_series, None)):

            t0 = time.perf_counter()
            _handle_X_y_dask(X, y)
            tf = time.perf_counter()

            print(f"x_idx {x_idx}, y_idx {y_idx}    t = {tf-t0} sec")







































