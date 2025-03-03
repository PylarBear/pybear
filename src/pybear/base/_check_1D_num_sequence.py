# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional
import numpy.typing as npt
from typing_extensions import TypeAlias, Union

import numbers

import numpy as np
import pandas as pd
import polars as pl
import dask.array as da
import dask.dataframe as ddf

from ..utilities._nan_masking import nan_mask
from ..utilities._inf_masking import inf_mask


PythonTypes: TypeAlias = Union[list, tuple, set]
NumpyTypes: TypeAlias = npt.NDArray
PandasTypes: TypeAlias = pd.Series
PolarsTypes: TypeAlias = pl.Series
DaskTypes: TypeAlias = Union[da.Array, ddf.Series]

XContainer: TypeAlias = \
    Union[PythonTypes, NumpyTypes, PandasTypes, PolarsTypes]




def check_1D_num_sequence(
    num_sequence:XContainer[numbers.Number],
    require_all_finite:Optional[bool] = False
) -> None:

    """
    Validate things that are expected to be 1D sequences of numbers.
    Accepts 1D python built-ins, numpy arrays, pandas series, and polars
    series. When 'require_all_finite' is True, every element in the
    sequence must be an instance of numbers.Number; if there is a
    nan-like or infinity-like value a ValueError will be raised. If
    'require_all_finite' is False, non-finite values are ignored and only
    the finite values must be an instance of numbers.Number. If all
    checks pass then None is returned.


    Parameters
    ----------
    num_sequence:
        XContainer[numbers.Number] - something that is expected to be a
        1D sequence of numbers.
    require_all_finite:
        Optional[bool], default=False - if True, disallow all non-finite
        values, such as nan-like or infinity-like values.


    Raises
    ------
    TypeError:
        for invalid container
    ValueError:
        for non-finite values when 'require_all_finite' is True


    Return
    ------
    -
        None


    Notes
    -----
    Type Aliases

    PythonTypes: Union[list, tuple, set]

    NumpyTypes: npt.NDArray

    PandasTypes: pd.Series

    PolarsTypes: pl.Series

    DaskTypes: Union[da.Array, ddf.Series]

    XContainer: Union[PythonTypes, NumpyTypes, PandasTypes, PolarsTypes]


    Examples
    --------
    >>> from pybear.base import check_1D_num_sequence
    >>> X = [1, 2, 3, np.nan, 4]
    >>> check_1D_num_sequence(X, require_all_finite=False)

    >>> try:
    ...     check_1D_num_sequence(X, require_all_finite=True)
    ... except ValueError as e:
    ...     print(e)
    got non-finite values when not allowed

    """


    _err_msg = f"Expected a 1D sequence of number-like values. "
    _addon = (
        f"\nAccepted containers are python lists, tuples, and sets, "
        f"\nnumpy 1D arrays, pandas series, polars series, dask 1D "
        f"\narrays, and dask series."
    )


    # block disallowed containers -- -- -- -- -- -- -- -- -- -- -- -- --
    if hasattr(num_sequence, 'toarray'):
        raise TypeError(_err_msg + _addon)
    if isinstance(num_sequence, (pd.DataFrame, pl.DataFrame, ddf.DataFrame)):
        raise TypeError(_err_msg + _addon)

    try:
        # must be iterable
        iter(num_sequence)
        # cant be string or dict
        if isinstance(num_sequence, (str, dict)):
            raise Exception
        # handle dask or anything with shape attr directly
        if len(getattr(num_sequence, 'shape', [1])) != 1:
            raise Exception
        # inside cant have iterables, including strings
        for __ in num_sequence:
            try:
                iter(__)
                raise UnicodeError
            except UnicodeError:
                raise Exception
            except Exception as e:
                continue
    except:
        raise TypeError(_err_msg + _addon)

    del _addon
    # END block disallowed containers -- -- -- -- -- -- -- -- -- -- --


    # we have a 1D that has no strings or iterables
    # it may have junky values like pd.NA

    # need to know this whether or not disallowing non-finite
    _non_finite_mask = nan_mask(num_sequence).astype(np.uint8)
    _non_finite_mask += inf_mask(num_sequence).astype(np.uint8)
    _non_finite_mask = _non_finite_mask.astype(bool)
    if not any(_non_finite_mask):
        # if its all false, save the memory
        _non_finite_mask = []

    # check for finiteness
    if require_all_finite and any(_non_finite_mask):
        raise ValueError(f"got non-finite values when not allowed")

    # if we get to here, we do not have non-finite or are allowing

    # avoid a copy if we can
    if not any(_non_finite_mask):
        if not all(map(
            isinstance,
            num_sequence,
            (numbers.Number for i in num_sequence)
        )):
            raise TypeError(_err_msg)
    else:
        _finite = np.array(list(num_sequence))[np.logical_not(_non_finite_mask)]
        if not all(map(isinstance, _finite, (numbers.Number for i in _finite))):
            raise TypeError(_err_msg)
        del _finite

    del _err_msg, _non_finite_mask




