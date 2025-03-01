# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import XContainer

from ....utilities._nan_masking import nan_mask

from .._validation._X import _val_X



def _transform(
    _X: XContainer,
    _new_value: any
) -> XContainer:

    """
    Map new values to the nan-like representations in X. Cannot be scipy
    sparse dok or lil, it must have a 'data' attribute.


    Parameters
    ----------
    _X:
        Union[NDArray, pandas.Series, pandas.DataFrame, polars.Series,
        polars.DataFrame, scipy.sparse], of shape (n_samples, n_features)
        or (n_samples,) - the object for which to replace nan-like
        representations.
    _new_value:
        any - the new value to put in place of the nan-like values.
        There is no validation for this value, the user is free to enter
        whatever they like. If there is a casting problem, i.e., the
        receiving object, the data, will not receive the given value,
        then any exceptions would be raised by the receiving object.


    Returns
    -------
    -
        _X:
            Union[NDArray, pandas.Series, pandas.DataFrame, polars.Series,
            polars.DataFrame, scipy.sparse] of shape (n_samples,
            n_features), (n_samples,), or (n_non_zera_values,) - The
            original data with new values in the locations previously
            occupied by nan-like values.

    """


    _val_X(_X)

    if hasattr(_X, 'toarray'):
        # for scipy, need to mask the 'data' attribute
        _X.data[nan_mask(_X.data)] = _new_value
    elif hasattr(_X, 'clone'):
        _og_type = type(_X)
        _X = _X.to_numpy().copy()
        _X[nan_mask(_X)] = _new_value
        _X = _og_type(_X)
        del _og_type
    else:
        _X[nan_mask(_X)] = _new_value


    return _X


