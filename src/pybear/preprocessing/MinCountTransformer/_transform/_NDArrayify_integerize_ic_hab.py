# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._validation._ign_cols_hab_callable import _val_ign_cols_hab_callable
from ....utilities._feature_name_mapper import feature_name_mapper

from typing_extensions import Union
import numpy.typing as npt
from .._type_aliases import (
    XContainer,
    IgnoreColumnsType,
    HandleAsBoolType,
    InternalIgnoreColumnsType,
    InternalHandleAsBoolType
)

from copy import deepcopy
import numpy as np



def _NDArrayify_integerize_ic_hab(
    X: XContainer,
    _ignore_columns: IgnoreColumnsType,
    _handle_as_bool: HandleAsBoolType,
    _n_features_in: int,
    _feature_names_in: Union[npt.NDArray[str], None]
) -> tuple[InternalIgnoreColumnsType, InternalHandleAsBoolType]:

    """
    Receive 'ignore_columns' and 'handle_as_bool' as callable(X),
    Iterable[str], Iterable[int], or None, and convert to Iterable[int]
    with all non-negative indices. Perform any validation that can be
    done against n_features_in and feature_names_in, if passed.
    This module is essentially a hub that centralizes calling any
    callables and validating, converting any None values to an empty
    ndarray, and converting any string values to indices.

    if ignore_columns and handle_as_bool were originally passed to MCT
    as Iterable[int] or Iterable[str], the dimensions of them or the
    feature names in them would have been validated in
    _validation > _val_ignore_columns_handle_as_bool. Here is where
    values passed as feature names are finally mapped to indices.


    """


    # ignore_columns  -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    if callable(_ignore_columns):

        if not hasattr(X, 'copy'):
            raise TypeError(f"'X' must have a 'copy' method")

        try:
            # pybear requires .copy()
            # protect from user-defined function mutating X
            __ignore_columns = _ignore_columns(X.copy())
        except Exception as e:
            raise Exception(
                f"ignore_columns callable excepted with {type(e)} error"
            ) from e

        _val_ign_cols_hab_callable(
            __ignore_columns,
            'ignore_columns',
            _n_features_in,
            _feature_names_in
        )
    elif _ignore_columns is None:
        __ignore_columns = np.array([], dtype=np.int32)
    else:
        __ignore_columns = deepcopy(_ignore_columns)

    __ignore_columns: InternalIgnoreColumnsType = \
        feature_name_mapper(
            __ignore_columns,
            _feature_names_in,
            positive=True
        )
    # END ignore_columns  -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    # handle_as_bool  -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    if callable(_handle_as_bool):

        if not hasattr(X, 'copy'):
            raise TypeError(f"'X' must have a 'copy' method")

        try:
            # pybear requires .copy()
            # protect from user-defined function mutating X
            __handle_as_bool = _handle_as_bool(X.copy())
        except Exception as e:
            raise Exception(
                f"ignore_columns callable excepted with {type(e)} error"
            ) from e

        _val_ign_cols_hab_callable(
            __handle_as_bool,
            'ignore_columns',
            _n_features_in,
            _feature_names_in
        )
    elif _handle_as_bool is None:
        __handle_as_bool = np.array([], dtype=np.int32)
    else:
        __handle_as_bool = deepcopy(_handle_as_bool)

    __handle_as_bool: InternalIgnoreColumnsType = \
        feature_name_mapper(
            __handle_as_bool,
            _feature_names_in,
            positive=True
        )
    # END handle_as_bool  -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    return __ignore_columns, __handle_as_bool















