# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

# pizza see if this can be combined with conflict_warner


from typing import Iterable
from typing_extensions import Union
from .._type_aliases import (
    InternalHandleAsBoolType,
    OriginalDtypesType
)

import warnings
import numbers
import numpy as np

from .._validation._original_dtypes import _val_original_dtypes
from .._validation._ignore_columns_handle_as_bool import \
    _val_ignore_columns_handle_as_bool



def _val_handle_as_bool_v_dtypes(
    _handle_as_bool: Union[Iterable[numbers.Integral], None],
    _ignore_columns: Union[Iterable[numbers.Integral], None],
    _original_dtypes: OriginalDtypesType,
    _raise: bool=False
) -> InternalHandleAsBoolType:

    """
    Validate that the columns to be handled as boolean are numeric
    columns (MCT internal dtypes 'bin_int', 'int', 'float'). MCT internal
    dtype 'obj' columns cannot be handled as boolean, and this module
    will raise it finds this condition and :param: '_raise' is True. If
    '_raise' is False, it will warn. If an 'obj' column that is in
    '_handle_as_bool' and is also in '_ignore_columns', '_ignore_columns'
    trumps '_handle_as_bool' and the column is ignored.

    '_handle_as_bool' must be received as a 1D list-like of integers or
    None. To be a list-like, this means 2 things must have happened
    before calling this function:

    1) if 'handle_as_bool' was a callable, it must have been computed
    already

    2) if the callable returned a vector of strings or a vector of
    strings was originally passed to MCT, it must have been converted to
    integer column indices already


    Parameters
    ----------
    _handle_as_bool:
        Union[Iterable[numbers.Integral], None] - the column indices to
        be handled as boolean, i.e., all zero values are handled as False
        and all non-zero values are handled as True.
    _ignore_columns:
        Union[Iterable[numbers.Integral], None] - the column indices to
        be ignored when applying the minimum frequency threshold.
    _original_dtypes:
        Iterable[Union[Literal['bin_int', 'int', 'float', 'obj']]] -
        The datatypes for each column in the dataset as determined by
        MCT. Values can be 'bin_int', 'int', 'float', or 'obj'.
    _raise:
        bool - If True, raise a ValueError if handle-as-bool columns are
        'obj' dtype; if False, emit a warning.


    Return
    ------
    -
        _handle_as_bool: npt.NDArray[np.int32] - The final list of the
        columns to be handled as boolean. If there was any intersection
        between the original _handle_as_bool indices and ignored indices,
        the ignored indices have been removed and this is a list of
        numerical columns to be handled as boolean.

    """


    if _handle_as_bool is None:
        return

    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    _val_original_dtypes(_original_dtypes)

    _val_ignore_columns_handle_as_bool(
        _handle_as_bool,
        'handle_as_bool',
        ['Iterable[int]', 'None'],
        _n_features_in=len(_original_dtypes),
        _feature_names_in=None
    )

    _val_ignore_columns_handle_as_bool(
        _ignore_columns,
        'ignore_columns',
        ['Iterable[int]', 'None'],
        _n_features_in=len(_original_dtypes),
        _feature_names_in=None
    )


    if not isinstance(_raise, bool):
        raise TypeError("'_raise' must be boolean")


    if len(_handle_as_bool) == 0:
        return np.array(list(_handle_as_bool), dtype=np.int32)

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    if _ignore_columns is not None:
        _ic_hab_intersection = \
            set(list(_ignore_columns)).intersection(list(_handle_as_bool))
    else:
        _ic_hab_intersection = set()

    _hab_not_ignored = list(set(_handle_as_bool) - set(_ic_hab_intersection))
    # MASK gives the _handle_as_bool columns that arent ignored
    _hab_not_ignored_dtypes = np.array(list(_original_dtypes))[_hab_not_ignored]

    if 'obj' in _hab_not_ignored_dtypes:
        MASK = (_hab_not_ignored_dtypes == 'obj')
        IDXS = ', '.join(map(str, np.array(list(_hab_not_ignored))[MASK]))

        _base_msg = (
            f"cannot use handle_as_bool on str/object columns "
            f"--- column index(es) == {IDXS}. "
        )

        if _raise:
            raise ValueError(_base_msg)
        else:
            _addon = (
                f"\na warning is emitted and exception is not raised. \nafter "
                f"fitting, you may be able to use set_params to correct this "
                f"issue. in :method: transform, this condition will raise an "
                f"exception."
            )
            warnings.warn(_base_msg + _addon)
            del _base_msg, _addon


    del _ic_hab_intersection

    return np.array(list(_hab_not_ignored), dtype=np.int32)






