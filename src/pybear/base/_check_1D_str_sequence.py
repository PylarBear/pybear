# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional, Iterable
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




def check_1D_str_sequence(
    str_sequence:XContainer[str],
    require_all_finite:Optional[bool] = False
) -> None:

    """
    Validate things that are expected to be 1D sequences of strings.
    Accepts 1D python built-ins, numpy arrays, pandas series, and
    polars series. When 'require_all_finite' is True, every element
    in the sequence must be an instance of str; if there is a nan-like
    or infinity-like value a ValueError will be raised.
    If 'require_all_finite' is False, non-finite values are ignored and
    only the finite values must be an instance of str. If all checks
    pass then None is returned.


    Parameters
    ----------
    str_sequence:
        XContainer[str] - something that is expected to be a 1D sequence
        of strings.
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
    >>> from pybear.base import check_1D_str_sequence
    >>> X = ['a', 'b', 'c', 'nan', 'd']
    >>> check_1D_str_sequence(X, require_all_finite=False)

    >>> try:
    ...     check_1D_str_sequence(X, require_all_finite=True)
    ... except ValueError as e:
    ...     print(e)
    got non-finite values when not allowed

    """


    _err_msg = f"Expected a 1D sequence of string values. "
    _addon = (
        f"\nAccepted containers are python lists, tuples, and sets, "
        f"\nnumpy 1D arrays, pandas series, polars series, dask 1D "
        f"\narrays, and dask series."
    )


    # block disallowed containers -- -- -- -- -- -- -- -- -- -- -- -- --
    if hasattr(str_sequence, 'toarray'):
        raise TypeError(_err_msg + _addon)
    if isinstance(str_sequence, (pd.DataFrame, pl.DataFrame, ddf.DataFrame)):
        raise TypeError(_err_msg + _addon)

    try:
        # must be iterable
        iter(str_sequence)
        # cant be string or dict
        if isinstance(str_sequence, (str, dict)):
            raise Exception
        # handle dask or anything with shape attr directly
        if len(getattr(str_sequence, 'shape', [1])) != 1:
            raise Exception
        # inside cant have non-string iterables, but it may have funky
        # junk like nans
        for __ in str_sequence:
            if isinstance(__, Iterable) and not isinstance(__, str):
                raise Exception
    except:
        raise TypeError(_err_msg + _addon)

    del _addon
    # END block disallowed containers -- -- -- -- -- -- -- -- -- -- --


    # we have a 1D that has no strings or iterables
    # it may have junky values like pd.NA

    # need to know this whether or not disallowing non-finite
    _non_finite_mask = nan_mask(str_sequence).astype(np.uint8)
    _non_finite_mask += inf_mask(str_sequence).astype(np.uint8)
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
            str_sequence,
            (str for i in str_sequence)
        )):
            raise TypeError(_err_msg)
    else:
        _finite = np.array(list(str_sequence))[np.logical_not(_non_finite_mask)]
        if not all(map(isinstance, _finite, (str for i in _finite))):
            raise TypeError(_err_msg)
        del _finite

    del _err_msg, _non_finite_mask




