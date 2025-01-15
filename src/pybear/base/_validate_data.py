# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from ._cast_to_ndarray import cast_to_ndarray as _cast_to_ndarray
from ._check_is_finite import check_is_finite
from ._check_scipy_sparse import check_scipy_sparse
from ._ensure_2D import ensure_2D as _ensure_2D
from ._check_dtype import check_dtype
from ._check_shape import check_shape
from ._set_order import set_order

from typing import Literal, Iterable
from typing_extensions import Union

import numbers

import numpy as np




def validate_data(
    X,
    *,
    copy_X:bool=True,
    cast_to_ndarray:bool=False,
    accept_sparse:Iterable[Literal[
        "csr", "csc", "coo", "dia", "lil", "dok", "bsr"
    ]]=("csr", "csc", "coo", "dia", "lil", "dok", "bsr"),
    dtype:Literal['numeric','any']='any',
    require_all_finite:bool=True,
    cast_inf_to_nan:bool=True,
    standardize_nan:bool=True,
    allowed_dimensionality:Iterable[numbers.Integral]=(1,2),
    ensure_2d:bool=True,
    order:Literal['C', 'F']='C',
    ensure_min_features:numbers.Integral=1,
    ensure_max_features:Union[numbers.Integral, None]=None,
    ensure_min_samples:numbers.Integral=1,
    sample_check:Union[numbers.Integral, None]=None
):

    """
    Validate characteristics of X and apply some select transformative
    operations. This module is intended for validation of X in methods of
    pybear estimators and transformers, but can be used in stand-alone
    applications.

    All the functionality carried out in this module is executed by
    individual modules, that is, this module is basically a central hub
    that unifies all the separate operations. Some of the individual
    modules may have particular requirements of X such as a specific
    container like a numpy array, or that the container expose methods
    like 'copy' or attributes like 'shape'. See the individual modules
    for specifics.

    This module can perform many checks and transformative operations in
    preparation for pybear estimators or transformers. See the Parameters
    section for an exhaustive list of the functionality.


    Parameters
    ----------
    X:
        array-like of shape (n_samples, n_features) or (n_samples,) -
        the data to be validated.
    copy_X:
        bool, default=True - whether to operate directly on the passed X
        or create a copy.
    cast_to_ndarray:
        bool, default=False - if True, convert the passed X to numpy
        ndarray.
    accept_sparse:
        Union[Iterable[Literal["csr", "csc", "coo", "dia", "lil", "dok",
        "bsr"]], False, None], default=("csr", "csc", "coo", "dia",
        "lil", "dok", "bsr") - The allowed scipy sparse matrix/array
        formats. If no scipy sparse are allowed, False or None can be
        passed, and an exception will be raised if X is a scipy sparse
        object. Otherwise, must be a 1D vector-like (such as a python
        list or tuple) containing some or all of the 3-character acronyms
        shown here. Not case sensitive. Entries cover both the 'matrix'
        and 'array' formats, e.g., ['csr', 'csc'] will allow csr_matrix,
        csr_array, csc_matrix, and csc_array formats.
    dtype:
        Literal['numeric','any'], default='any' - the allowed datatype
        of X. If 'numeric', data that cannot be coerced to a numeric
        datatype will raise an exception. If 'any', no restrictions are
        imposed on the datatype of X.
    require_all_finite:
        bool, default=True - if True, block data that has undefined
        values, in particular, nan-like and infinity-like values. If
        False, nan-like and infinity like values are allowed.
    cast_inf_to_nan:
        bool, default=True - If True, coerce any infinity-like values in
        the data to numpy.nan; if False, leave any infinity-like values
        as is.
    standardize_nan:
        bool, default=True - If True, coerce all nan-like values in the
        data to numpy.nan; if False, leave all the nan-like values in
        the given state.
    allowed_dimensionality:
        Iterable[numbers.Integral] - The allowed dimensionalities of
        X. All entries must be greater than zero and less than or
        equal to two. Examples: (1,)  {1,2}, [2]
    ensure_2d:
        bool, default=True - coerce the data to a 2-dimensional format.
        For example, a 1D numpy vector would be reshaped to a 2D numpy
        array; a 1D pandas series would be converted to a 2D pandas
        dataframe.
    order:
        Literal['C', 'F'], default='C' - only applicable if X is a numpy
        array or :param: cast_to_ndarray is True. Sets the memory order
        of X. 'C' is row-major and 'F' is column-major. The default
        for numpy arrays is 'C', and major packages like scikit-learn
        typically expect to see numpy arrays with 'C' order. pybear
        recommends that this parameter be used with understanding of the
        potential performance implications of changing the memory order
        of X on downstream processes that may be designed for 'C' order.
    ensure_min_features:
        numbers.Integral, default=1 - the minimum number of features
        (columns) that must be in X.
    ensure_max_features:
        Union[numbers.Integral, None] - The maximum number of features
        allowed in X; if not None, must be greater than or equal to
        ensure_min_features. If None, then there is no restriction on
        the maximum number of features in X.
    ensure_min_samples:
        numbers.Integral, default=1 - the minimum number of samples
        (rows) that must be in X. Ignored if :param: sample_check is not
        None.
    sample_check:
        Union[numbers.Integral, None] - The exact number of samples
        allowed in OBJECT. If not None, must be a non-negative integer.
        Use this to check, for example, that the number of samples in y
        equals the number of samples in X. If None, this check is not
        performed.


    Return
    ------
    -
        X: the validated data.

    """

    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # no validation for X! the entire module is for validation of X!

    # copy_X -- -- -- -- -- -- -- -- -- -- -- --
    if not isinstance(copy_X, bool):
        raise TypeError(f"'copy_X' must be boolean.")
    # END copy_X -- -- -- -- -- -- -- -- -- -- --

    # cast_to_ndarray -- -- -- -- -- -- -- -- -- --
    if not isinstance(cast_to_ndarray, bool):
        raise TypeError(f"'cast_to_ndarray' must be boolean.")
    # END cast_to_ndarray -- -- -- -- -- -- -- -- --

    # accept_sparse -- -- -- -- -- -- -- -- -- --
    err_msg = (f":param: 'accept_sparse' must be None, literal False, or a "
        f"vector-like iterable of literals indicating the types of scipy "
        f"sparse containers that are allowed. see the docs for the valid "
        f"literals accepted in the :param: 'accept_sparse' iterable.")

    try:
        if accept_sparse is None:
            raise UnicodeError
        if accept_sparse is False:
            raise UnicodeError
        iter(accept_sparse)
        if isinstance(accept_sparse, (str, dict)):
            raise Exception
        if not all(map(isinstance, accept_sparse, (str for _ in accept_sparse))):
            raise Exception
        accept_sparse = list(map(str.lower, accept_sparse))
        valid = ["csr", "csc", "coo", "dia", "lil", "dok", "bsr"]
        for _ in accept_sparse:
            if _ not in valid:
                raise MemoryError
        del valid
    except UnicodeError:
        pass
    except MemoryError:
        raise ValueError(err_msg)
    except:
        raise TypeError(err_msg)

    del err_msg
    # END accept_sparse -- -- -- -- -- -- -- -- --

    # dtype -- -- -- -- -- -- -- -- -- -- -- --
    err_msg = (
        f"'dtype' must be string literal 'numeric' or 'any', not case sensitive."
    )
    if not isinstance(dtype, str):
        raise TypeError(err_msg)
    dtype = dtype.lower()
    if not dtype in ['numeric', 'any']:
        raise ValueError(err_msg)
    del err_msg
    # END dtype -- -- -- -- -- -- -- -- -- -- --

    # require_all_finite -- -- -- -- -- -- -- -- -- -- -- --
    if not isinstance(require_all_finite, bool):
        raise TypeError(f"'require_all_finite' must be boolean.")
    # END require_all_finite -- -- -- -- -- -- -- -- -- -- --

    # cast_inf_to_nan -- -- -- -- -- -- -- -- -- -- -- --
    if not isinstance(cast_inf_to_nan, bool):
        raise TypeError(f"'cast_inf_to_nan' must be boolean.")
    # END cast_inf_to_nan -- -- -- -- -- -- -- -- -- -- --

    # standardize_nan -- -- -- -- -- -- -- -- -- -- -- --
    if not isinstance(standardize_nan, bool):
        raise TypeError(f"'standardize_nan' must be boolean.")
    # END standardize_nan -- -- -- -- -- -- -- -- -- -- --


    if require_all_finite and standardize_nan:
        raise ValueError(
            f"if :param: require_all_finite is True, then :param: "
            f"standardize_nan must be False."
        )

    if require_all_finite and cast_inf_to_nan:
        raise ValueError(
            f"if :param: require_all_finite is True, then :param: "
            f"cast_inf_to_nan must be False."
        )

    # allowed_dimensionality -- -- -- -- -- -- -- -- -- -- -- --
    __ = allowed_dimensionality
    err_msg = f"'allowed_dimensionality' must be a 1D iterable of positive integers."
    try:
        iter(__)
        if isinstance(__, (str, dict)):
            raise Exception
        if not all(map(isinstance, __, (numbers.Integral for _ in __))):
            raise Exception
        if not all(map(lambda x: x > 0, __)):
            raise UnicodeError
        if not all(map(lambda x: x < 3, __)):
            raise UnicodeError
    except UnicodeError:
        raise ValueError(err_msg)
    except:
        raise TypeError(err_msg)
    # END ensure_2d -- -- -- -- -- -- -- -- -- -- --

    # ensure_2d -- -- -- -- -- -- -- -- -- -- -- --
    if not isinstance(ensure_2d, bool):
        raise TypeError(f"'ensure_2d' must be boolean.")
    # END ensure_2d -- -- -- -- -- -- -- -- -- -- --

    # order -- -- -- -- -- -- -- -- -- -- -- -- -- --
    err_msg = f"'order' must be string literal 'C' or 'F', not case sensitive."
    if not isinstance(order, str):
        raise TypeError(err_msg)
    order = order.upper()
    if order not in ['C', 'F']:
        raise ValueError(err_msg)
    del err_msg
    # order -- -- -- -- -- -- -- -- -- -- -- -- -- --

    # ensure_min_features -- -- -- -- -- -- -- -- -- -- -- -- --
    err_msg = f"'ensure_min_features' must be a non-negative integer."
    if not isinstance(ensure_min_features,  numbers.Integral):
        raise TypeError(err_msg)
    if isinstance(ensure_min_features, bool):
        raise TypeError(err_msg)
    if not ensure_min_features >= 0:
        raise ValueError(err_msg)
    del err_msg
    # END ensure_min_features -- -- -- -- -- -- -- -- -- -- -- --

    # ensure_max_features -- -- -- -- -- -- -- -- -- -- -- -- --
    if ensure_max_features is not None:
        err_msg = (
            f"'ensure_max_features' must be None or a non-negative integer "
            f"greater than or equal to :param: 'ensure_min_features'."
        )
        if not isinstance(ensure_max_features,  numbers.Integral):
            raise TypeError(err_msg)
        if isinstance(ensure_max_features, bool):
            raise TypeError(err_msg)
        if ensure_max_features < ensure_min_features:
            raise ValueError(err_msg)
        del err_msg
    # END ensure_max_features -- -- -- -- -- -- -- -- -- -- -- --

    # ensure_min_samples / sample_check -- -- -- -- -- -- -- -- -- -- --
    if sample_check is None:
        err_msg = f"'ensure_min_features' must be a non-negative integer."
        if not isinstance(ensure_min_samples,  numbers.Integral):
            raise TypeError(err_msg)
        if isinstance(ensure_min_samples, bool):
            raise TypeError(err_msg)
        if not ensure_min_samples >= 0:
            raise ValueError(err_msg)
        del err_msg
    elif sample_check is not None:
        err_msg = (f"'sample_check' must be None or a non-negative integer.")
        if not isinstance(sample_check, numbers.Integral):
            raise TypeError(err_msg)
        if isinstance(sample_check, bool):
            raise TypeError(err_msg)
        if sample_check < 0:
            raise ValueError(err_msg)
        del err_msg
    # END ensure_min_samples / sample_check -- -- -- -- -- -- -- -- -- -

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *




    # avoid multiple copies of X. do not set 'copy_X' for each of the
    # functions to True! create only one copy of X, set copy_X to False
    # for all the functions.
    if copy_X:
        _X = X.copy()
    else:
        _X = X
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    # accept_sparse
    check_scipy_sparse(
        _X,
        allowed=accept_sparse
    )
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    if cast_to_ndarray:
        _X = _cast_to_ndarray(
            _X,
            copy_X=False
        )
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    if ensure_2d:
        _X = _ensure_2D(
            _X,
            copy_X=False
        )
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    if isinstance(_X, np.ndarray):
        _X = set_order(
            _X,
            order=order,
            copy_X=False
        )
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    if require_all_finite or cast_inf_to_nan or standardize_nan:

        # this must be before check_dtype to ensure that ndarrays have
        # only np.nans in them if standardize_nan is True. otherwise
        # an ndarray that is expected to have only np.nans in it will
        # fail a 'numeric' dtype check before the nans are standardized.

        _X = check_is_finite(
            _X,
            allow_nan=not require_all_finite,
            allow_inf=not require_all_finite,
            cast_inf_to_nan=cast_inf_to_nan,
            standardize_nan=standardize_nan,
            copy_X=False
        )
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    check_dtype(
        _X,
        allowed=dtype
    )
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    check_shape(
        _X,
        min_features=ensure_min_features,
        max_features=ensure_max_features,
        min_samples=ensure_min_samples,
        sample_check=sample_check,
        allowed_dimensionality=allowed_dimensionality
        # if n_features_in_ is 1, then dimensionality could be 1 or 2, 
        # for any number of features greater than 1 dimensionality must
        # be 2.
    )
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    return _X

















