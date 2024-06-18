# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing import Union, Iterable, TypeAlias
from ._type_aliases import DataType

import numpy as np

XType: TypeAlias = Iterable[Iterable[DataType]]
YType: TypeAlias = Iterable[Iterable[DataType]]


def _handle_X_y(
        X: XType,
        y: Union[YType, None],
        _name: str,  # use type(self).__name__ from MCT call
        __x_original_obj_type: Union[str, None], # use self._x_original_obj_type
        __y_original_obj_type: Union[str, None], # self._y_original_obj_type
    ) -> tuple[XType, YType, str, str, Union[np.ndarray[int], None]]:


    """
    Validate dimensions of X and y, get dtypes of first-seen data objects,
    standardize the containers for processing, get column names if available.
    
    Parameters
    ----------
        X:
            [ndarray, pandas.DataFrame, pandas.Series] - data object
        y:
            [ndarray, pandas.DataFrame, pandas.Series] - target object
        _name:
            str, use type(self).__name__ from MCT call
        __x_original_obj_type:
            Union[str, None], use self._x_original_obj_type
        __y_original_obj_type:
            Union[str, None], self._y_original_obj_type
    
    Return
    ------
    -
        X: ndarray - The given data as ndarray.

        y: ndarray - The given target as ndarray.

        __x_original_obj_type: Union[str, None] - validated object type

        __y_original_obj_type: Union[str, None] - validated object type

        _columns: ndarray[str] - Feature names extracted from X.

    """

    # THE INTENT IS TO RUN EVERYTHING AS NP ARRAY TO STANDARDIZE INDEXING.
    # IF DFS ARE PASSED, COLUMNS CAN OPTIONALLY BE PULLED OFF AND RETAINED.


    _columns = None

    # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
    # IF X IS Series, MAKE DataFrame

    # _x_original_obj_type ONLY MATTERS WHEN _handle_X_y IS CALLED
    # THROUGH transform() (OR fit_transform())

    # validate og_obj_dtypes ** * ** * ** * ** * ** * ** * ** * ** * **

    _base_err = (f"must be in [None, 'numpy_array', 'pandas_dataframe', "
                 f"'pandas_series']")
    err_msg_x = f"__x_original_obj_type " + _base_err
    err_msg_y = f"__y_original_obj_type " + _base_err

    _allowed = [None, 'numpy_array', 'pandas_dataframe', 'pandas_series']

    try:
        if __x_original_obj_type is not None:
            __x_original_obj_type = __x_original_obj_type.lower()
    except:
        raise TypeError(err_msg_x)

    try:
        if __y_original_obj_type is not None:
            __y_original_obj_type = __y_original_obj_type.lower()
    except:
        raise TypeError(err_msg_y)

    if __x_original_obj_type not in _allowed:
        raise ValueError(err_msg_x)

    if __y_original_obj_type not in _allowed:
        raise ValueError(err_msg_y)

    del _base_err, err_msg_x, err_msg_y
    # END validate og_obj_dtypes ** * ** * ** * ** * ** * ** * ** * ** *


    try:
        X = X.to_frame()
        __x_original_obj_type = __x_original_obj_type or 'pandas_series'
    except:
        pass

    # X COULD BE np OR pdDF
    try:
        _columns = np.array(X.columns)
    except:
        pass

    # IF pdDF CONVERT TO np
    try:
        X = X.to_numpy()
        __x_original_obj_type = __x_original_obj_type or 'pandas_dataframe'
    except:
        pass

    # IF daskDF CONVERT TO np
    try:
        X = X.to_dask_array().compute()
        __x_original_obj_type = __x_original_obj_type or 'pandas_dataframe'
    except:
        pass


    if isinstance(X, np.recarray):
        raise TypeError(f"{_name} cannot take numpy recarrays. "
            f"Pass X as a numpy.ndarray, pandas dataframe, or pandas series.")

    if isinstance(X, np.ma.core.MaskedArray):
        raise TypeError(f"{_name} cannot take numpy masked arrays. "
            f"Pass X as a numpy.ndarray, pandas dataframe, or pandas series.")

    if not isinstance(X, (type(None), str, dict)):
        try:
            list(X[:10])
            _dtype = np.array(X).dtype
            if any([_ in str(_dtype).lower() for _ in ['int', 'float']]):
                X = np.array(X, dtype=_dtype)
            else:
                X = np.array(X, dtype=object)
            del _dtype
        except:
            pass

    if not isinstance(X, np.ndarray):
        raise TypeError(f"X is an invalid data-type {type(X)}")

    __x_original_obj_type = __x_original_obj_type or 'numpy_array'

    if len(X.shape) == 1:
        X = X.reshape((-1, 1))

    # *** X MUST BE np ***

    #  ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    # self._y_original_obj_type ONLY MATTERS WHEN _handle_X_y IS
    # CALLED THROUGH transform() (OR fit_transform())

    if y is not None:
        # IF y IS Series, MAKE DataFrame
        try:
            y = y.to_frame()
            __y_original_obj_type = __y_original_obj_type or 'pandas_series'
        except:
            pass

        # y COULD BE np OR pdDF
        # IF pdDF, CONVERT TO np
        try:
            y = y.to_numpy()
            __y_original_obj_type = __y_original_obj_type or 'pandas_dataframe'
        except:
            pass

        # IF daskDF CONVERT TO np
        try:
            y = y.to_dask_array().compute()
            __y_original_obj_type = __y_original_obj_type or 'pandas_dataframe'
        except:
            pass

        if isinstance(y, np.recarray):
            raise TypeError(f"{_name} cannot take numpy recarrays. "
                            f"Pass y as a numpy.ndarray.")

        if isinstance(y, np.ma.core.MaskedArray):
            raise TypeError(
                f"{_name} cannot take numpy masked arrays. Pass y as a "
                f"numpy.ndarray, pandas dataframe, or pandas series.")

        try:
            y = np.array(y)
        except:
            pass

        if not isinstance(y, np.ndarray):
            raise TypeError(f"y is an unknown data-type {type(y)}")

        __y_original_obj_type = __y_original_obj_type or 'numpy_array'

        # WORKS IF len(y.shape) IS 1 or 2
        _X_rows, _y_rows = X.shape[0], y.shape[0]
        if _X_rows != _y_rows:
            raise ValueError(f"the number of rows in y ({_y_rows}) does "
                             f"not match the number of rows in X ({_X_rows})")

    # *** y MUST BE np OR None ***


    return X, y, __x_original_obj_type, __y_original_obj_type, _columns





















