# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

# pizza see if this can be combined with handle_as_bool_v_dtypes
# pizza what about when threshold==1? same as ignoring, right?

from .._type_aliases import (
    OriginalDtypesType,
    InternalIgnoreColumnsType,
    InternalHandleAsBoolType
)
from typing_extensions import Union

import warnings
import numpy as np

from .._validation._n_features_in import _val_n_features_in
from .._validation._original_dtypes import _val_original_dtypes
from .._validation._ignore_columns_handle_as_bool import _val_ignore_columns_handle_as_bool



def _conflict_warner(
    _original_dtypes: OriginalDtypesType,
    _handle_as_bool: Union[InternalHandleAsBoolType, None],
    _ignore_columns: Union[InternalIgnoreColumnsType, None],
    _ignore_float_columns: bool,
    _ignore_non_binary_integer_columns: bool,
    _n_features_in: int
) -> None:

    """
    Determine if there is any intersection between columns to be handled
    as bool and any of the ignored columns. There is a hierarchy of
    what takes precedence, ignored columns always supersede handling as
    boolean.

    If handle_as_bool is None or an empty list, bypass all of this.


    Parameters
    ----------
    _original_dtypes:
        npt.NDArray[Union[Literal['bin_int', 'int', 'float', 'obj']]] -
        the datatypes assigned to the data by MCT.
    _ignore_columns:
        npt.NDArray[np.int32] - the column indices to be excluded from
        the thresholding rules.
    _handle_as_bool:
        npt.NDArray[np.int32] - the column indices to be handled as
        boolean, i.e, 0 is handled as False and anything non-zero is
        handled as True. MCT internal datatype 'obj' columns cannot be
        handled as boolean.
    _ignore_float_columns:
        bool - whether to exclude float columns from the thresholding
        rules.
    _ignore_non_binary_integer_columns:
        bool - whether to exclude non-binary integer columns from the
        thresholding rules.
    _n_features_in:
        int - the number of features in the data.


    Return
    ------
    -
        None


    """

    # basic validation ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    _val_n_features_in(_n_features_in)

    _val_original_dtypes(_original_dtypes)

    assert len(_original_dtypes) == _n_features_in

    assert isinstance(_ignore_non_binary_integer_columns, bool)

    _val_ignore_columns_handle_as_bool(
        _ignore_columns,
        'ignore_columns',
        ['Iterable[int]', 'None'],
        _n_features_in=_n_features_in,
        _feature_names_in=None
    )

    _val_ignore_columns_handle_as_bool(
        _handle_as_bool,
        'handle_as_bool',
        ['Iterable[int]', 'None'],
        _n_features_in=_n_features_in,
        _feature_names_in=None
    )

    # END basic validation ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    if _handle_as_bool is None or len(_handle_as_bool) == 0:
        return


    # if intersection between ignore_float_columns and ignore_columns, doesnt
    # matter, ignored either way

    # if intersection between ignore_non_binary_integer_columns and ignore_columns,
    # doesnt matter, ignored either way

    # so we need to address when handle_as_bool intersects ignore_columns,
    # ignore_float_columns, and ignore_non_binary_integer_columns.

    if _ignore_float_columns:
        _float_columns = \
            np.arange(_n_features_in)[np.array(_original_dtypes) == 'float']

        __ = list(map(str, set(_handle_as_bool).intersection(_float_columns)))
        if len(__):
            q = "index" if len(__) == 1 else "indices"
            z = "is" if len(__) == 1 else "are"
            warnings.warn(
                f"column {q} {', '.join(__)} {z} designated as handle a bool "
                f"but {z} float and float columns are ignored. \nignore "
                f"float columns supersedes and the column is ignored. "
            )
            del q, z

        del _float_columns, __


    if _ignore_non_binary_integer_columns:
        _non_bin_int_columns = \
            np.arange(_n_features_in)[np.array(_original_dtypes) == 'int']

        __ = list(map(str, set(_handle_as_bool).intersection(_non_bin_int_columns)))
        if len(__):
            q = "index" if len(__) == 1 else "indices"
            z = "is" if len(__) == 1 else "are"
            warnings.warn(
                f"column {q} {', '.join(__)} {z} designated as handle a bool "
                f"but {z} non-binary integer and non-binary integer columns are "
                f"ignored. \nignore non-binary integer columns supersedes and "
                f"the column is ignored."
            )
            del q, z

        del _non_bin_int_columns, __


    if _ignore_columns is None:
        return

    if len(_ignore_columns):

        __ = list(map(str, set(_handle_as_bool).intersection(_ignore_columns)))
        if len(__):
            q = "index" if len(__) == 1 else "indices"
            z = "is" if len(__) == 1 else "are"
            warnings.warn(
                f"column {q} {', '.join(__)} {z} designated as handle a bool "
                f"but {z} is also in 'ignore_columns'. \n'ignore_columns' "
                f"supersedes and the column is ignored."
            )
            del q, z

        del __









