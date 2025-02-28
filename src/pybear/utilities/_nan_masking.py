# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union, TypeAlias
import numpy.typing as npt

import numbers

import numpy as np
import pandas as pd
import scipy.sparse as ss
import polars as pl



NumpyTypes: TypeAlias = npt.NDArray[numbers.Real]

PandasTypes: TypeAlias = Union[pd.DataFrame, pd.Series]

PolarsTypes: TypeAlias = Union[pl.DataFrame, pl.Series]

# dok and lil are left out intentionally
SparseTypes: TypeAlias = Union[
    ss._csr.csr_matrix, ss._csc.csc_matrix, ss._coo.coo_matrix,
    ss._dia.dia_matrix, ss._bsr.bsr_matrix, ss._csr.csr_array,
    ss._csc.csc_array, ss._coo.coo_array, ss._dia.dia_array,
    ss._bsr.bsr_array
]



def nan_mask_numerical(
    obj: Union[NumpyTypes, PandasTypes, PolarsTypes, SparseTypes]
) -> npt.NDArray[bool]:

    """
    This function accepts numpy arrays, pandas dataframes and series,
    polars dataframes and series, and all scipy sparse matrices/arrays
    except dok and lil formats. Data must be able to cast to numpy
    numerical dtypes. In all cases, return a boolean numpy array or
    vector indicating the locations of nan-like representations in the
    data. "nan-like representations" include, at least, numpy.nan,
    pandas.NA, None, and string representations of "nan". In the cases
    of numpy, pandas, and polars objects of shape (n_samples, n_features)
    or (n_samples, ), return an identically shaped numpy array. In the
    cases of scipy sparse objects, return a boolean numpy vector of shape
    equal to that of the 'data' attribute of the sparse object.


    TypeAliases
    -----------
    NumpyTypes: npt.NDArray[numbers.Real]

    PandasTypes: Union[pd.DataFrame, pd.Series]

    PolarsTypes: Union[pl.DataFrame, pl.Series]

    SparseTypes: Union[ss._csr.csr_matrix, ss._csc.csc_matrix,
        ss._coo.coo_matrix, ss._dia.dia_matrix, ss._bsr.bsr_matrix,
        ss._csr.csr_array, ss._csc.csc_array, ss._coo.coo_array,
        ss._dia.dia_array, ss._bsr.bsr_array]


    Parameters
    ----------
    obj:
        Union[NDArray, pandas.Series, pandas.DataFrame, polars.Series,
        polars.DataFrame, scipy.sparse] of shape (n_samples, n_features)
        or (n_samples, ) - the object for which to locate nan-like
        representations.


    Return
    ------
    mask:
        NDArray[bool] of shape (n_samples, n_features) or (n_samples, )
        or (n_non_zero_values, ), indicating nan-like representations in
        'obj' via the value boolean True. Values that are not nan-like
        are False.

    """

    _err_msg = (
        f"'obj' must be an array-like with a copy() or clone() method, "
        f"such as numpy arrays, scipy sparse matrices or arrays, pandas "
        f"dataframes/series, polars dataframes/series. \nif passing a "
        f"scipy sparse object, it cannot be dok or lil. \npython built-in "
        f"iterables, such as lists, sets, and tuples, are not allowed."
    )

    try:
        iter(obj)
        if isinstance(obj, (str, dict, list, tuple, set)):
            raise Exception
        if not hasattr(obj, 'copy') and not hasattr(obj, 'clone'):
            # copy for numpy, pandas, and scipy; clone for polars
            raise Exception
        if hasattr(obj, 'toarray'):
            if not hasattr(obj, 'data'): # ss dok
                raise Exception
            elif all(map(isinstance, obj.data, (list for _ in obj.data))):
                # ss lil
                raise Exception
            else:
                obj = obj.data
    except:
        raise TypeError(_err_msg)


    try:
        _ = obj.copy()  # numpy, pandas, and scipy
    except:
        # Polars uses zero-copy conversion when possible, meaning the
        # underlying memory is still controlled by Polars and marked
        # as read-only. NumPy and Pandas may inherit this read-only
        # flag, preventing modifications.
        # THE ORDER IS IMPORTANT HERE. CONVERT TO PANDAS FIRST, THEN COPY.
        _ = obj.to_pandas().copy()  # polars

    try:
        _ = _.to_numpy()
    except:
        pass

    _[(_ == 'nan')] = np.nan

    return pd.isna(_)



def nan_mask_string(
    obj: Union[NumpyTypes, PandasTypes, PolarsTypes]
) -> npt.NDArray[bool]:

    """
    This function accepts numpy arrays, pandas dataframes and series,
    and polars dataframes and series. In all cases, return an identically
    shaped boolean numpy array or vector indicating the locations of
    nan-like representations in the data. "nan-like representations"
    include, at least, pandas.NA, None (of type None, not string "None"),
    and string representations of "nan". This function does not accept
    scipy sparce matrices or arrays, as dok and lil formats are not
    handled globally in the nan_mask functions, and the remaining sparse
    objects cannot contain non-numeric data.


    TypeAliases
    -----------
    NumpyTypes: npt.NDArray[numbers.Real]

    PandasTypes: Union[pd.DataFrame, pd.Series]

    PolarsTypes: Union[pl.DataFrame, pl.Series]


    Parameters
    ----------
    obj:
        Union[NDArray, pandas.Series, pandas.DataFrame, polars.Series,
        polars.DataFrame] of shape (n_samples, n_features) or
        (n_samples, ) - the object for which to locate nan-like
        representations.


    Return
    ------
    mask:
        NDArray[bool] of shape (n_samples, n_features) or (n_samples),
        indicating nan-like representations in 'obj' via the value
        boolean True. Values that are not nan-like are False.


    """

    _err_msg = (
        f"'obj' must be an array-like with a copy() or clone() method, "
        f"such as numpy arrays, pandas dataframes/series, or polars "
        f"dataframes/series. \n'obj' cannot be a scipy sparse matrix or "
        f"array. \npython built-in iterables, such as lists, sets, and "
        f"tuples, are not allowed."
    )


    try:
        iter(obj)
        if isinstance(obj, (str, dict, list, set, tuple)):
            raise Exception
        if not hasattr(obj, 'copy') and not hasattr(obj, 'clone'):
            # copy for numpy, pandas, and scipy; clone for polars
            raise Exception
        if hasattr(obj, 'toarray'):
            raise Exception
    except:
        raise TypeError(_err_msg)

    try:
        _ = obj.copy()  # numpy, pandas, and scipy
    except:
        # Polars uses zero-copy conversion when possible, meaning the
        # underlying memory is still controlled by Polars and marked
        # as read-only. NumPy and Pandas may inherit this read-only
        # flag, preventing modifications.
        # Tests did not expose this as a problem like it did for numerical().
        # just to be safe though, do this the same way as numerical().
        _ = obj.to_pandas().copy()  # polars

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
    obj: Union[NumpyTypes, PandasTypes, PolarsTypes, SparseTypes]
) -> npt.NDArray[bool]:

    """
    This function combines the functionality of nan_mask_numerical and
    nan_mask_string, giving a centralized location for masking numerical
    and non-numerical data.

    For full details, see the docs for nan_mask_numerical and
    nan_mask_string.

    Briefly, when passing numerical or non-numerical data, this function
    accepts numpy arrays, pandas dataframes/series, and polars
    dataframes/series of shape (n_samples, n_features) or (n_samples, )
    and returns an identically sized numpy array of booleans indicating
    the locations of nan-like representations. Also, when passing
    numerical data, this function accepts scipy sparse matrices / arrays
    of all formats except dok and lil. In that case, a numpy boolean
    vector of shape identical to that of the sparse object's 'data'
    attribute is returned. "nan-like representations" include, at least,
    np.nan, pandas.NA, None (of type None, not string "None"), and string
    representations of "nan".


    TypeAliases
    -----------
    NumpyTypes: npt.NDArray[numbers.Real]

    PandasTypes: Union[pd.DataFrame, pd.Series]

    PolarsTypes: Union[pl.DataFrame, pl.Series]

    SparseTypes: Union[ss._csr.csr_matrix, ss._csc.csc_matrix,
        ss._coo.coo_matrix, ss._dia.dia_matrix, ss._bsr.bsr_matrix,
        ss._csr.csr_array, ss._csc.csc_array, ss._coo.coo_array,
        ss._dia.dia_array, ss._bsr.bsr_array]


    Parameters
    ----------
    obj:
        Union[NDArray, pandas.Series, pandas.DataFrame, polars.Series,
        polars.DataFrame, scipy.sparse], of shape (n_samples, n_features),
        (n_samples,), or (n_non_zero_values,)- the object for which to
        locate nan-like representations.


    Return
    ------
    mask:
        NDArray[bool] of shape (n_samples, n_features), (n_samples,) or
        (n_non_zero_values, ), indicating the locations of nan-like
        representations in 'obj' via the value boolean True. Values that
        are not nan-like are False.

    """


    if isinstance(obj, (str, dict, list, set, tuple)):
        raise TypeError(
            f"python built-in iterables, such as lists, sets, and "
            f"tuples, are not allowed."
        )

    try:
        if hasattr(obj, 'astype'):  # numpy, pandas, and scipy
            obj.astype(np.float64)
        elif hasattr(obj, 'cast'):  # polars
            obj.cast(pl.Float64)
            # if did not except
            raise TimeoutError
        else:
            raise NotImplementedError

        if isinstance(obj,
            (ss.dok_matrix, ss.lil_matrix, ss.dok_array, ss.lil_array)
        ):
            raise UnicodeError

        # numpy, pandas, and scipy
        return nan_mask_numerical(obj.astype(np.float64))

    except NotImplementedError:
        raise TypeError(f"invalid type {type(obj)} in nan_mask")
    except TimeoutError:
        # polars -- do this out from under the try in case this excepts
        return nan_mask_numerical(obj.cast(pl.Float64))
    except UnicodeError:
        raise TypeError(f"'obj' cannot be scipy sparse dok or lil")
    except:
        return nan_mask_string(obj)










