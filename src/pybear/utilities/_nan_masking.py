# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union, TypeAlias
import numpy.typing as npt

from copy import deepcopy

import numpy as np
import pandas as pd
import scipy.sparse as ss
import polars as pl



PythonTypes: TypeAlias = Union[list, tuple, set]

NumpyTypes: TypeAlias = npt.NDArray

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
    obj: Union[PythonTypes, NumpyTypes, PandasTypes, PolarsTypes, SparseTypes]
) -> npt.NDArray[bool]:

    """
    This function accepts python lists, tuples, and sets, numpy arrays,
    pandas dataframes and series, polars dataframes and series, and all
    scipy sparse matrices/arrays except dok and lil formats. It does not
    accept any ragged python built-ins, dask objects, numpy recarrays,
    or numpy masked arrays. Data must be able to cast to numpy numerical
    dtypes. In all cases, return a boolean numpy array or vector
    indicating the locations of nan-like representations in the data.
    "nan-like representations" include, at least, numpy.nan, pandas.NA,
    None, and string representations of "nan". In the cases of python
    native, numpy, pandas, and polars objects of shape (n_samples,
    n_features) or (n_samples, ), return an identically shaped numpy
    array. In the cases of scipy sparse objects, return a boolean numpy
    vector of shape equal to that of the 'data' attribute of the sparse
    object.


    Parameters
    ----------
    obj:
        Union[PythonTypes, NumpyTypes, PandasTypes, PolarsTypes,
        SparseTypes] of shape (n_samples, n_features) or (n_samples, ) -
        the object for which to locate nan-like representations.


    Return
    ------
    mask:
        NDArray[bool] of shape (n_samples, n_features) or (n_samples, )
        or (n_non_zero_values, ), indicating nan-like representations in
        'obj' via the value boolean True. Values that are not nan-like
        are False.


    Notes
    -----
    Type aliases

    PythonTypes: Union[list, tuple, set]

    NumpyTypes: npt.NDArray

    PandasTypes: Union[pd.DataFrame, pd.Series]

    PolarsTypes: Union[pl.DataFrame, pl.Series]

    SparseTypes: Union[ss._csr.csr_matrix, ss._csc.csc_matrix,
        ss._coo.coo_matrix, ss._dia.dia_matrix, ss._bsr.bsr_matrix,
        ss._csr.csr_array, ss._csc.csc_array, ss._coo.coo_array,
        ss._dia.dia_array, ss._bsr.bsr_array]


    Examples
    --------
    >>> from pybear.utilities import nan_mask_numerical
    >>> X = np.arange(6).astype(np.float64)
    >>> X[1] = np.nan
    >>> X[-2] = np.nan
    >>> X
    array([ 0., nan,  2.,  3., nan,  5.])
    >>> nan_mask_numerical(X)
    array([False,  True, False, False,  True, False])

    """


    # validation -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    _err_msg = (
        f"'obj' must be an array-like with a copy() or clone() method, "
        f"such as python built-ins, \nnumpy arrays, scipy sparse matrices "
        f"or arrays, pandas dataframes/series, polars dataframes/series. "
        f"\nif passing a scipy sparse object, it cannot be dok or lil. "
        f"\nnumpy recarrays and masked arrays are not allowed."
    )

    try:
        iter(obj)
        if isinstance(obj, (str, dict)):
            raise Exception
        if isinstance(obj, tuple):
            # tuple doesnt have copy() method, but is OK
            # notice the elif
            pass
        elif not hasattr(obj, 'copy') and not hasattr(obj, 'clone'):
            # copy for builtins, numpy, pandas, and scipy; clone for polars
            raise Exception
        if isinstance(obj, (np.recarray, np.ma.MaskedArray)):
            raise Exception
        if hasattr(obj, 'toarray'):
            if not hasattr(obj, 'data'): # ss dok
                raise Exception
            elif all(map(isinstance, obj.data, (list for _ in obj.data))):
                # ss lil
                raise Exception
    except:
        raise TypeError(_err_msg)

    del _err_msg

    if isinstance(obj, (list, tuple, set)):
        _err_msg = (
            f"nan_mask_numerical expected all number-like values. "
            f"\ngot at least one non-nan string."
        )
        if any(map(isinstance, obj, (str for i in obj))):
            # cant have strings except str(nan)
            if any(
                map(lambda x: x.lower() != 'nan',
                [i for i in obj if isinstance(i, str)])
            ):
                raise TypeError(_err_msg)

        try:
            if all(map(lambda x: x.lower() == 'nan', obj)):
                raise Exception
            # above we proved obj isnt a 1D of strings
            # this will except if obj is not 2D
            map(iter, obj)
            # prove not ragged
            if len(set(map(len, obj))) != 1:
                raise UnicodeError
            # we have a non-ragged 2D of somethings
            for row in obj:
                if any(map(isinstance, row, (str for i in row))):
                    if any(
                        map(lambda x: x.lower() != 'nan',
                        [i for i in row if isinstance(i, str)])
                    ):
                        raise TimeoutError
            # have a non-ragged 2D of non-strings
        except UnicodeError:
            raise ValueError(
                f"nan_mask_numerical does not accept ragged arrays"
            )
        except TimeoutError:
            raise TypeError(_err_msg)
        except Exception as e:
            # we have a 1D that has no strings, at least.
            # could have junky nans or None, or whatever else
            pass
        del _err_msg
    # END validation -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    # cant use pybear.base.copy_X here, circular import
    if hasattr(obj, 'toarray'):
        _ = obj.data.copy()
    elif hasattr(obj, 'clone'):
        # Polars uses zero-copy conversion when possible, meaning the
        # underlying memory is still controlled by Polars and marked
        # as read-only. NumPy and Pandas may inherit this read-only
        # flag, preventing modifications.
        # THE ORDER IS IMPORTANT HERE. CONVERT TO PANDAS FIRST, THEN COPY.
        _ = obj.to_pandas().copy()  # polars
    elif isinstance(obj, (list, tuple, set)):
        try:
            if all(map(lambda x: x.lower == 'nan', obj)):
                raise Exception
            # mapping list to obj is OK, if obj is 1D it cant have strings
            _ = list(map(list, deepcopy(obj)))
        except Exception as e:
            _ = list(deepcopy(obj))

        _ = np.array(_).astype(np.float64)
    else:
        _ = obj.copy()  # numpy, pandas, and scipy


    try:
        _ = _.to_numpy()
    except:
        pass

    _[(_ == 'nan')] = np.nan

    return pd.isna(_)



def nan_mask_string(
    obj: Union[PythonTypes, NumpyTypes, PandasTypes, PolarsTypes]
) -> npt.NDArray[bool]:

    """
    This function accepts python lists, tuples, and sets, numpy arrays,
    pandas dataframes and series, and polars dataframes and series. It
    does not accept any ragged python built-ins, dask objects, numpy
    recarrays, or numpy masked arrays. In all cases, return an
    identically shaped boolean numpy array or vector indicating the
    locations of nan-like representations in the data. "nan-like
    representations" include, at least, pandas.NA, None (of type None,
    not string "None"), and string representations of "nan". This
    function does not accept scipy sparce matrices or arrays, as dok and
    lil formats are not handled globally in the nan_mask functions, and
    the remaining sparse objects cannot contain non-numeric data.


    Parameters
    ----------
    obj:
        Union[PythonTypes, NumpyTypes, PandasTypes, PolarsTypes] of shape
        (n_samples, n_features) or (n_samples, ) - the object for which
        to locate nan-like representations.


    Return
    ------
    mask:
        NDArray[bool] of shape (n_samples, n_features) or (n_samples),
        indicating nan-like representations in 'obj' via the value
        boolean True. Values that are not nan-like are False.


    Notes
    -----
    Type aliases

    PythonTypes: Union[list, tuple, set]

    NumpyTypes: npt.NDArray

    PandasTypes: Union[pd.DataFrame, pd.Series]

    PolarsTypes: Union[pl.DataFrame, pl.Series]


    Examples
    --------
    >>> from pybear.utilities import nan_mask_string
    >>> X = list('abcde')
    >>> X[0] = 'nan'
    >>> X[2] = 'nan'
    >>> X
    ['nan', 'b', 'nan', 'd', 'e']
    >>> nan_mask_string(X)
    array([ True, False,  True, False, False])

    """

    # validation -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    _err_msg = (
        f"'obj' must be an array-like with a copy() or clone() method, "
        f"\nsuch as python built-ins, pandas dataframes/series, numpy "
        f"arrays, or polars dataframes/series. \n'obj' cannot be a scipy "
        f"sparse matrix or array. \nnumpy recarrays and masked arrays "
        f"are not allowed."
    )

    try:
        iter(obj)
        if isinstance(obj, (str, dict)):
            raise Exception
        if isinstance(obj, tuple):
            pass
            # tuple doesnt have copy() method
            # notice the elif
        elif not hasattr(obj, 'copy') and not hasattr(obj, 'clone'):
            # copy for numpy, pandas, and scipy; clone for polars
            raise Exception
        if isinstance(obj, (np.recarray, np.ma.MaskedArray)):
            raise Exception
        if hasattr(obj, 'toarray'):
            raise Exception
    except:
        raise TypeError(_err_msg)

    del _err_msg

    if isinstance(obj, (list, set, tuple)):

        try:
            if all(map(isinstance, obj, (str for i in obj))):
                raise Exception
            # this will except if obj is not 2D, because cant be all strings
            map(iter, obj)
            # prove not ragged
            if len(set(map(len, obj))) != 1:
                raise UnicodeError
            # we have a non-ragged 2D of somethings
        except UnicodeError:
            raise ValueError(
                f"nan_mask_string does not accept ragged arrays"
            )
        except Exception as e:
            # we have a 1D list-like of strings
            pass

    # END validation -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    if hasattr(obj, 'clone'):
        # Polars uses zero-copy conversion when possible, meaning the
        # underlying memory is still controlled by Polars and marked
        # as read-only. NumPy and Pandas may inherit this read-only
        # flag, preventing modifications.
        # Tests did not expose this as a problem like it did for numerical().
        # just to be safe though, do this the same way as numerical().
        _ = obj.to_pandas().copy()  # polars
    elif isinstance(obj, (list, tuple, set)):
        # we cant just map list here, if 1D it is full of strings
        # if one is str, assume all entries are not list-like
        # what about non-str nans
        if any(map(isinstance, obj, (str for i in obj))):
            _ = list(deepcopy(obj))
        elif any(map(lambda x: str(x).lower() == 'nan', obj)):
            _ = list(deepcopy(obj))
        elif any(map(lambda x: x is None, obj)):
            _ = list(deepcopy(obj))
        else:
            # otherwise, assume all entries are list-like
            _ = list(map(list, deepcopy(obj)))

        _ = np.array(_)
    else:
        _ = obj.copy()  # numpy, pandas, and scipy

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
    obj: Union[PythonTypes, NumpyTypes, PandasTypes, PolarsTypes, SparseTypes]
) -> npt.NDArray[bool]:

    """
    This function combines the functionality of nan_mask_numerical and
    nan_mask_string, giving a centralized location for masking numerical
    and non-numerical data.

    For full details, see the docs for nan_mask_numerical and
    nan_mask_string.

    Briefly, when passing numerical or non-numerical data, this function
    accepts python built-ins, numpy arrays, pandas dataframes/series,
    and polars dataframes/series of shape (n_samples, n_features) or
    (n_samples, ) and returns an identically sized numpy array of
    booleans indicating the locations of nan-like representations. Also,
    when passing numerical data, this function accepts scipy sparse
    matrices / arrays of all formats except dok and lil. In that case,
    a numpy boolean vector of shape identical to that of the sparse
    object's 'data' attribute is returned. "nan-like representations"
    include, at least, np.nan, pandas.NA, None (of type None, not string
    "None"), and string representations of "nan". This function does not
    accept any ragged python built-ins, dask objects, numpy recarrays,
    or numpy masked arrays.


    Parameters
    ----------
    obj:
        Union[PythonTypes, NumpyTypes, PandasTypes, PolarsTypes,
        SparseTypes], of shape (n_samples, n_features), (n_samples,), or
        (n_non_zero_values,) - the object for which to locate nan-like
        representations.


    Return
    ------
    mask:
        NDArray[bool] of shape (n_samples, n_features), (n_samples,) or
        (n_non_zero_values, ), indicating the locations of nan-like
        representations in 'obj' via the value boolean True. Values that
        are not nan-like are False.


    Notes
    -----
    PythonTypes: Union[list, tuple, set]

    NumpyTypes: npt.NDArray

    PandasTypes: Union[pd.DataFrame, pd.Series]

    PolarsTypes: Union[pl.DataFrame, pl.Series]

    SparseTypes: Union[ss._csr.csr_matrix, ss._csc.csc_matrix,
        ss._coo.coo_matrix, ss._dia.dia_matrix, ss._bsr.bsr_matrix,
        ss._csr.csr_array, ss._csc.csc_array, ss._coo.coo_array,
        ss._dia.dia_array, ss._bsr.bsr_array]


    Examples
    --------
    >>> from pybear.utilities import nan_mask
    >>> X1 = np.arange(6).astype(np.float64)
    >>> X1[0] = np.nan
    >>> X1[-1] = np.nan
    >>> X1
    array([nan,  1.,  2.,  3.,  4., nan])
    >>> nan_mask(X1)
    array([ True, False, False, False, False,  True])

    >>> X2 = list('vwxyz')
    >>> X2[0] = 'nan'
    >>> X2[2] = 'nan'
    >>> X2
    ['nan', 'w', 'nan', 'y', 'z']
    >>> nan_mask(X2)
    array([ True, False,  True, False, False])

    """


    if isinstance(obj, (str, dict)):
        raise TypeError(f"only list-like or array-like objects are allowed.")

    try:
        if isinstance(obj, (list, tuple, set)):
            pd.Series(list(obj)).to_numpy().astype(np.float64)
            raise IndexError
        elif hasattr(obj, 'astype'):  # numpy, pandas, and scipy
            if isinstance(obj,
                (ss.dok_matrix, ss.lil_matrix, ss.dok_array, ss.lil_array)
            ):
                raise UnicodeError
            obj.astype(np.float64)
            raise MemoryError
        elif hasattr(obj, 'cast'):  # polars
            obj.cast(pl.Float64)
            # if did not except
            raise TimeoutError
        else:
            raise NotImplementedError

    except UnicodeError:
        raise TypeError(f"'obj' cannot be scipy sparse dok or lil")
    except NotImplementedError:
        raise TypeError(f"invalid type {type(obj)} in nan_mask")
    except IndexError:
        # do this out from under the try in case this excepts
        return nan_mask_numerical(obj)
    except MemoryError:
        # do this out from under the try in case this excepts
        return nan_mask_numerical(obj.astype(np.float64))
    except TimeoutError:
        # polars -- do this out from under the try in case this excepts
        return nan_mask_numerical(obj.cast(pl.Float64))
    except:
        return nan_mask_string(obj)










