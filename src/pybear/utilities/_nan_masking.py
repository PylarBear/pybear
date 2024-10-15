# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import numpy as np
import pandas as pd

from typing_extensions import Union
import numpy.typing as npt



def nan_mask_numerical(
    obj: Union[npt.NDArray[float], pd.DataFrame]
) -> npt.NDArray[bool]:

    """
    Return a boolean numpy array of shape (n_samples, n_features)
    indicating the locations of nan-like representations in a numerical
    numpy ndarray or pandas dataframe of shape (n_samples, n_features).
    "nan-like representations" include, at least, numpy.nan, pandas.NA,
    and string representations of "nan".


    Parameters
    ----------
    obj:
        Union[npt.NDArray[float], pd.DataFrame[Union[float, int]], of
        shape (n_samples, n_features) - the object for which to locate
        nan-like representations.


    Return
    ------
    mask:
        NDArray[bool], of shape (n_samples, n_features), indicating
        nan-like representations in 'obj' via the value boolean True.
        Values that are not nan-like are False.

    """

    _err_msg = (f"'obj' must be an array-like with a copy() method, such as "
                f"numpy array or pandas dataframe")

    try:
        iter(obj)
        if isinstance(obj, (str, dict)):
            raise Exception
        if not hasattr(obj, 'copy'):
            raise Exception
    except:
        raise TypeError(_err_msg)


    _ = obj.copy()

    try:
        _ = _.to_numpy()
    except:
        pass

    _[(_ == 'nan')] = np.nan

    return pd.isna(_)



def nan_mask_string(
    obj: Union[npt.NDArray[str], pd.DataFrame]
) -> npt.NDArray[bool]:

    """
    Return a boolean numpy array of shape (n_samples, n_features)
    indicating the locations of nan-like representations in a string-type
    or object-type numpy ndarray or pandas dataframe of shape (n_samples,
    n_features). "nan-like representations" include, at least, pandas.NA,
    None (of type None, not string 'None'), and string representations
    of "nan".


    Parameters
    ----------
    obj:
        Union[npt.NDArray[str], pd.DataFrame[str], of shape (n_samples,
        n_features) - the object for which to locate nan-like
        representations.


    Return
    ------
    mask:
        NDArray[bool], of shape (n_samples, n_features), indicating
        nan-like representations in 'obj' via the value boolean True.
        Values that are not nan-like are False.

    """

    _err_msg = (f"'obj' must be an array-like with a copy() method, such as "
                f"numpy array or pandas dataframe")

    try:
        iter(obj)
        if isinstance(obj, (str, dict)):
            raise Exception
        if not hasattr(obj, 'copy'):
            raise Exception
    except:
        raise TypeError(_err_msg)

    _ = obj.copy()

    try:
        _[pd.isna(_)] = 'nan'
    except:
        pass

    try:
        _ = _.to_numpy()
    except:
        pass

    _ = np.char.replace(_.astype(str), None, 'nan')

    _ = np.char.replace(_, '<NA>', 'nan')

    _ = np.char.upper(_)

    return (_ == 'NAN').astype(bool)


def nan_mask(
    obj: Union[npt.NDArray[Union[float, str]], pd.DataFrame]
) -> npt.NDArray[bool]:

    """
    Return a boolean numpy array of shape (n_samples, n_features)
    indicating the locations of nan-like representations in a numerical
    or categorical numpy ndarray or pandas dataframe of shape (n_samples,
    n_features). "nan-like representations" include, at least, np.nan,
    pandas.NA, None (of type None, not string 'None'), and string
    representations of "nan".


    Parameters
    ----------
    obj:
        Union[npt.NDArray[Union[float, str]], pd.DataFrame[Union[float,
        int, str]], of shape (n_samples, n_features) - the object for
        which to locate nan-like representations.


    Return
    ------
    mask:
        NDArray[bool], of shape (n_samples, n_features), indicating
        nan-like representations in 'obj' via the value boolean True.
        Values that are not nan-like are False.

    """

    try:
        obj.astype(np.float64)
        return nan_mask_numerical(obj.astype(np.float64))
    except:
        return nan_mask_string(obj)










