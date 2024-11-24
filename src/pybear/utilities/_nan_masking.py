# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import numpy as np
import pandas as pd
import scipy.sparse as ss

from typing_extensions import Union, TypeAlias
import numpy.typing as npt



SparseTypes: TypeAlias = Union[
    ss._csr.csr_matrix,
    ss._csc.csc_matrix,
    ss._coo.coo_matrix,
    ss._dia.dia_matrix,
    ss._bsr.bsr_matrix,
    ss._csr.csr_array,
    ss._csc.csc_array,
    ss._coo.coo_array,
    ss._dia.dia_array,
    ss._bsr.bsr_array
]




def nan_mask_numerical(
    obj: Union[npt.NDArray, pd.DataFrame, SparseTypes]
) -> npt.NDArray[bool]:

    """
    This formula accepts numerical numpy arrays, pandas dataframes, and
    all scipy sparse matrices/arrays except dok and lil formats. In all
    cases, return a boolean numpy array or vector indicating the locations
    of nan-like representations in the data. "nan-like representations"
    include, at least, numpy.nan, pandas.NA, and string representations
    of "nan".
    In the cases of numpy arrays and pandas dataframes of shape
    (n_samples, n_features), return an identically shaped numpy array.
    In the cases of scipy sparse objects, return a boolean numpy vector
    of shape equal to that of the 'data' attribute of the sparse object.


    Parameters
    ----------
    obj:
        NDArray[float, int], pandas.DataFrame[float, int], or
        scipy.sparse[float, int], of shape (n_samples, n_features) - the
        object for which to locate nan-like representations.


    Return
    ------
    mask:
        NDArray[bool], of shape (n_samples, n_features) or of shape
        (n_non_zero_values, ), indicating nan-like representations in
        'obj' via the value boolean True. Values that are not nan-like
        are False.

    """

    _err_msg = (
        f"'obj' must be an array-like with a copy() method, such as numpy "
        f"array, pandas dataframe, or scipy sparse matrix or array. if "
        f"passing a scipy sparse object, it cannot be dok or lil."
    )

    try:
        iter(obj)
        if isinstance(obj, (str, dict)):
            raise Exception
        if not hasattr(obj, 'copy'):
            raise Exception
        if hasattr(obj, 'toarray'):
            if not hasattr(obj, 'data'): # ss dok
                raise Exception
            elif all(map(isinstance, obj.data, (list for _ in obj.data))): # ss lil
                raise Exception
            else:
                obj = obj.data
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
    of "nan". This function does not accept scipy sparce matrices or
    arrays, as dok and lil formats are not handled globally in the
    nan_mask functions, and the remaining sparse objects cannot contain
    non-numeric data.


    Parameters
    ----------
    obj:
        NDArray[str] or pandas.DataFrame[str] of shape (n_samples,
        n_features) - the object for which to locate nan-like
        representations.


    Return
    ------
    mask:
        NDArray[bool], of shape (n_samples, n_features), indicating
        nan-like representations in 'obj' via the value boolean True.
        Values that are not nan-like are False.

    """

    _err_msg = (
        f"'obj' must be an array-like with a copy() method, such as numpy "
        f"array or pandas dataframe. 'obj' cannot be a scipy sparse matrix "
        f"or array."
    )

    try:
        iter(obj)
        if isinstance(obj, (str, dict)):
            raise Exception
        if not hasattr(obj, 'copy'):
            raise Exception
        if hasattr(obj, 'toarray'):
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

    _ = np.char.replace(np.array(_).astype(str), None, 'nan')

    _ = np.char.replace(_, '<NA>', 'nan')

    _ = np.char.upper(_)

    return (_ == 'NAN').astype(bool)


def nan_mask(
    obj: Union[npt.NDArray, pd.DataFrame, SparseTypes]
) -> npt.NDArray[bool]:

    """
    This function combines the functionality of nan_mask_numerical and
    nan_mask_string.

    For full details, see the docs for nan_mask_numerical and
    nan_mask_string.

    Briefly, when passing numerical or non-numerical data, this function
    accepts numpy arrays or pandas dataframes of shape (n_samples,
    n_features) and returns an identically sized numpy array of booleans
    indicating the locations of nan-like representations. Also, when
    passing numerical data, this function accepts scipy sparse matrices /
    arrays of all formats except dok and lil. In that case, a numpy
    boolean vector of shape identical to that of the sparse object's
    'data' attribute is returned. "nan-like representations" include, at
    least, np.nan, pandas.NA, None (of type None, not string 'None'),
    and string representations of "nan".


    Parameters
    ----------
    obj:
        NDArray[any], pandas.DataFrame[any], or scipy.sparse[float, int],
        of shape (n_samples, n_features) - the object for which to locate
        nan-like representations.


    Return
    ------
    mask:
        NDArray[bool], of shape (n_samples, n_features) or of shape
        (n_non_zero_values, ), indicating nan-like representations in
        'obj' via the value boolean True. Values that are not nan-like
        are False.

    """

    try:
        obj.astype(np.float64)
        if isinstance(obj,
            (ss.dok_matrix, ss.lil_matrix, ss.dok_array, ss.lil_array)
        ):
            raise UnicodeError
        return nan_mask_numerical(obj.astype(np.float64))
    except UnicodeError:
        raise TypeError(f"'obj' cannot be scipy sparse dok or lil")
    except:
        return nan_mask_string(obj)










