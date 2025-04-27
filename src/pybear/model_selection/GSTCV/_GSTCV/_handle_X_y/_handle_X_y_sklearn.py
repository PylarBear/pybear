# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from ..._type_aliases import (
    XInputType,
    YInputType,
    XSKWIPType,
    YSKWIPType,
    FeatureNamesInType
)

import numpy as np
import pandas as pd



def _handle_X_y_sklearn(
    X: XInputType,
    y: YInputType = None
) -> tuple[XSKWIPType, YSKWIPType, FeatureNamesInType, int]:

    """
    Process given X and y into numpy ndarrays. All SK GSTCV internals
    require ndarrays. Accepts objects that can be converted to numpy
    arrays, including pandas dataframes and series. y, if 2 dimensional,
    is converted to a 1 dimensional vector.

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
            NDArray[Union[int, float]] - the given X array converted to
            numpy array.
        _y:
            NDArray[Union[int, float]] - the given y vector / array
            converted to a numpy ndarray vector.
        _feature_names_in:
            Union[NDArray[str], None] - if an object that has column
            names, such as a pandas dataframe or series, is passed, the
            column names are extracted and returned as a numpy vector.
            Otherwise, None is returned.
        _n_features_in:
            int - the number of columns in X.



    """


    err_msg = lambda _name, _object: (f"{_name} was passed with unknown "
        f"data type '{type(_object)}'. Use numpy array, pandas series, "
        f"or pandas dataframe.")

    _feature_names_in = None

    _X = X
    _y = y

    try:
        X_shape = _X.shape
    except:
        raise TypeError(err_msg('X', _X))

    # X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    if isinstance(_X, np.ndarray):
        pass

    elif isinstance(_X, (pd.Series, pd.DataFrame)):
        try:
            _X = _X.to_frame()
        except:
            pass

        _feature_names_in = np.array(_X.columns)

        _X = _X.to_numpy()

    else:
        raise TypeError(err_msg('X', _X))

    if len(X_shape) == 1:
        _X = _X.reshape((X_shape[0], 1))
    elif len(X_shape) == 2:
        pass
    else:
        raise ValueError(f"'X' must be a 1 or 2 dimensional object")

    X_shape = _X.shape

    _n_features_in = X_shape[1]

    # try to convert _X to np.uint8, to prove only numerical
    try:
        _X.astype(np.uint8)
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
        pass
    elif isinstance(_y, (pd.Series, pd.DataFrame)):
        try:
            _y = _y.to_frame()
        except:
            pass

        _y = _y.to_numpy()
    else:
        raise TypeError(err_msg('y', _y))


    if _y is not None:

        y_shape = _y.shape

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
                raise ValueError(
                    f"Classification metrics can't handle a mix of "
                    f"multilabel-indicator and binary targets"
                )
        else:
            raise ValueError(f"'y' must be a 1 or 2 dimensional object")

        if not set(np.unique(_y)).issubset({0, 1}):
            raise ValueError(
                f"GSTCV can only perform thresholding on binary targets "
                f"with values in [0,1]. Pass 'y' as a vector of 0's and "
                f"1's."
            )


    return _X, _y, _feature_names_in, _n_features_in








