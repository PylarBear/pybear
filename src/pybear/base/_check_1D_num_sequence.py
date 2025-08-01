# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import (
    Iterable,
    Optional
)
from typing_extensions import (
    TypeAlias,
    Union
)
from .__type_aliases import (
    Python1DTypes,
    Numpy1DTypes,
    Pandas1DTypes,
    Polars1DTypes
)

import numbers

import numpy as np
import pandas as pd
import polars as pl

from ..utilities._nan_masking import nan_mask
from ..utilities._inf_masking import inf_mask



XContainer: TypeAlias = \
    Union[Python1DTypes, Numpy1DTypes, Pandas1DTypes, Polars1DTypes]



def check_1D_num_sequence(
    X:XContainer[numbers.Number],
    require_all_finite:Optional[bool] = False
) -> None:
    """Validate things that are expected to be 1D sequences of numbers.

    Accepts 1D Python built-ins, numpy arrays, pandas series, and
    polars series. When `require_all_finite` is True, every element in
    the sequence must be an instance of numbers.Number; a ValueError
    will be raised if there are any nan-like or infinity-like values.
    If `require_all_finite` is False, non-finite values are ignored
    and only the finite values must be an instance of numbers.Number.
    If all checks pass then None is returned.

    Parameters
    ----------
    X : XContainer[numbers.Number] of shape (n_samples,)
        Something that is expected to be a 1D sequence of numbers.
    require_all_finite : Optional[bool], default=False
        If True, disallow all non-finite values, such as nan-like or
        infinity-like values.

    Raises
    ------
    TypeError:
        For invalid container.
    ValueError:
        For non-finite values when `require_all_finite` is True.

    Returns
    -------
    None

    Notes
    -----

    **Type Aliases**

    Python1DTypes:
        Union[list, tuple, set]

    Numpy1DTypes:
        numpy.ndarray

    Pandas1DTypes:
        pandas.core.series.Series

    Polars1DTypes:
        polars.series.Series

    XContainer:
        Union[Python1DTypes, Numpy1DTypes, Pandas1DTypes, Polars1DTypes]

    Examples
    --------
    >>> from pybear.base import check_1D_num_sequence
    >>> X = [1, 2, 3, np.nan, 4]
    >>> check_1D_num_sequence(X, require_all_finite=False)
    >>> try:
    ...     check_1D_num_sequence(X, require_all_finite=True)
    ... except ValueError as e:
    ...     print(e)
    Got non-finite values when not allowed.

    """


    _err_msg = f"Expected a 1D sequence of number-like values. "
    _addon = (
        f"\nAccepted containers are python lists, tuples, and sets, "
        f"numpy 1D arrays, pandas series, polars series."
    )


    # block disallowed containers -- -- -- -- -- -- -- -- -- -- -- -- --
    if hasattr(X, 'toarray'):
        raise TypeError(_err_msg + _addon)
    if isinstance(X, (pd.DataFrame, pl.DataFrame)):
        raise TypeError(_err_msg + _addon)

    try:
        # must be iterable
        iter(X)
        # cant be string or dict
        if isinstance(X, (str, dict)):
            raise Exception
        # handle anything with shape attr directly
        if hasattr(X, 'shape'):
            if len(getattr(X, 'shape')) != 1:
                raise Exception
            raise UnicodeError
        # inside cant have non-string iterables, but it may have funky
        # junk like nans
        # dont validate if there are non-num here, let that get picked
        # up at the bottom of the module so it sends the correct error
        for __ in X:
            if isinstance(__, Iterable) and not isinstance(__, str):
                raise Exception
    except UnicodeError:
        pass
    except Exception as e:
        raise TypeError(_err_msg + _addon)

    del _addon
    # END block disallowed containers -- -- -- -- -- -- -- -- -- -- --


    # we have a 1D that has no strings or iterables
    # it may have junky values like pd.NA

    # need to know this whether or not disallowing non-finite
    _non_finite_mask = nan_mask(X).astype(np.uint8)
    _non_finite_mask += inf_mask(X).astype(np.uint8)
    _non_finite_mask = _non_finite_mask.astype(bool)
    if not np.any(_non_finite_mask):
        # if its all false, save the memory
        _non_finite_mask = []

    # check for finiteness
    if require_all_finite and np.any(_non_finite_mask):
        raise ValueError(f"Got non-finite values when not allowed.")

    # if we get to here, we do not have non-finite or are allowing

    # avoid a copy if we can
    if not np.any(_non_finite_mask):
        if not all(map(
            isinstance,
            X,
            (numbers.Number for i in X)
        )):
            raise TypeError(_err_msg)
    else:

        _finite = np.array(list(X))[np.logical_not(_non_finite_mask)]

        try:
            _finite = _finite.astype(np.float64)
        except:
            pass

        if not all(map(
            isinstance,
            _finite,
            (numbers.Number for i in _finite)
        )):
            raise TypeError(_err_msg)

        del _finite

    del _err_msg, _non_finite_mask




