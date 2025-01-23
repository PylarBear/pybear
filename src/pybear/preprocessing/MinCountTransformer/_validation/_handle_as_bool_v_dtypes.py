# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing import Iterable
from typing_extensions import Union
from .._type_aliases import (
    OriginalDtypesType,
    InternalHandleAsBoolType
)

import numbers
import numpy as np


# pizza maybe move this?
# since it is not just val anymore, now mutates and returns.
# pizza proofread this and test and make some decisions

# pizza, think on this.
# this is expected to see _handle_as_bool and _ignore_columns after fully
# conditioned, i.e., as np.ndarray.astype(np.int32), which would be
# InternalHandleAsBoolType and InternalIgnoreColumnsType. But, that full
# constraint is not imposed here, only requires Iterable[int]. BUT...
# original_dtypes is in the same situation, is expected to be np.ndarray,
# and that requirement is imposed, i.e., cannot be list, set, tuple, can
# only be ndarray. decide what u want to do here.


def _val_handle_as_bool_v_dtypes(
    _handle_as_bool: Union[Iterable[numbers.Integral], None],
    _ignore_columns: Union[Iterable[numbers.Integral], None],
    _original_dtypes: OriginalDtypesType
) -> InternalHandleAsBoolType:

    """
    Validate that the columns to be handled as boolean are numeric
    columns (MCT internal dtypes 'bin_int', 'int', 'float'). MCT internal
    dtype 'obj' columns cannot be handled as boolean. If an 'obj' column
    that is in '_handle_as_bool' and is also in '_ignore_columns',
    '_ignore_columns' trumps '_handle_as_bool' and the column is ignored.

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
        npt.NDArray[Union[Literal['bin_int', 'int', 'float', 'obj']]] -
        The datatypes for each column in the dataset as determined by
        MCT. Values can be 'bin_int', 'int', 'float', or 'obj'.


    Return
    ------
    -
        _handle_as_bool: InternalHandleAsBoolType

    """


    if _handle_as_bool is None:
        return

    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    # handle_as_bool -- -- -- -- -- -- -- -- -- --
    _err_msg = f"'_handle_as_bool' must be a 1D list-like of integers or None"

    try:
        iter(_handle_as_bool)
        if isinstance(_handle_as_bool, (str, dict)):
            raise Exception
        if not len(np.array(list(_handle_as_bool)).shape) == 1:
            raise Exception
        if not all(map(
            isinstance,
            _handle_as_bool,
            (numbers.Integral for _ in _handle_as_bool)
        )):
            raise Exception
    except:
        raise TypeError(_err_msg)
    del _err_msg
    # END handle_as_bool -- -- -- -- -- -- -- -- -- --

    # ignore_columns -- -- -- -- -- -- -- -- -- --
    _err_msg = f"'_ignore_columns' must be a 1D list-like of integers or None"

    try:
        if _ignore_columns is None:
            raise UnicodeError
        iter(_ignore_columns)
        if isinstance(_ignore_columns, (str, dict)):
            raise Exception
        if not len(np.array(list(_ignore_columns)).shape) == 1:
            raise Exception
        if not all(map(
            isinstance,
            _ignore_columns,
            (numbers.Integral for _ in _ignore_columns)
        )):
            raise Exception
    except UnicodeError:
        pass
    except:
        raise TypeError(_err_msg)
    del _err_msg
    # END ignore_columns -- -- -- -- -- -- -- -- -- --

    # original_dtypes -- -- -- -- -- -- -- -- -- --
    _allowed = ['bin_int', 'int', 'float', 'obj']
    _err_msg = (
        f"'_original_dtypes' must be a 1D numpy ndarray of values in "
        f"{', '.join(_allowed)}."
    )
    try:
        if not isinstance(_original_dtypes, np.ndarray):
            raise Exception
        if not len(_original_dtypes.shape) == 1:
            raise Exception
        if not all(map(
            isinstance, _original_dtypes, (str for _ in _original_dtypes)
        )):
            raise Exception
        for _ in _original_dtypes:
            if _ not in _allowed:
                _addon = f"got '{_}'"
                raise UnicodeError
    except UnicodeError:
        raise ValueError(_err_msg + _addon)
    except:
        raise TypeError(_err_msg)
    del _err_msg, _allowed
    # END original_dtypes -- -- -- -- -- -- -- -- -- --

    if len(_handle_as_bool) == 0:
        return np.array(_handle_as_bool, dtype=np.int32)

    # joint -- -- -- -- -- -- -- -- -- -- -- -- -- --
    _n_features_in = len(_original_dtypes)

    if min(_handle_as_bool) < -_n_features_in:
        raise ValueError(
            f"'handle_as_bool' index {min(_handle_as_bool)} is out of bounds "
            f"for data with {_n_features_in} features"
        )
    if max(_handle_as_bool) >= _n_features_in:
        raise ValueError(
            f"'handle_as_bool' index {max(_handle_as_bool)} is out of bounds "
            f"for data with {_n_features_in} features"
        )

    if _ignore_columns is not None and len(_ignore_columns) > 0:
        if min(_ignore_columns) < -_n_features_in:
            raise ValueError(
                f"'ignore_columns' index {min(_ignore_columns)} is out of bounds "
                f"for data with {_n_features_in} features"
            )
        if max(_ignore_columns) >= _n_features_in:
            raise ValueError(
                f"'ignore_columns' index {max(_ignore_columns)} is out of bounds "
                f"for data with {_n_features_in} features"
            )
    # END joint -- -- -- -- -- -- -- -- -- -- -- -- -- --

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    if _ignore_columns is not None:
        _ic_hab_intersection = \
            set(list(_ignore_columns)).intersection(list(_handle_as_bool))
    else:
        _ic_hab_intersection = set()

    _hab_not_ignored = list(set(_handle_as_bool) - set(_ic_hab_intersection))
    # MASK gives the _handle_as_bool columns that arent ignored
    _hab_not_ignored_dtypes = _original_dtypes[_hab_not_ignored]

    if 'obj' in _hab_not_ignored_dtypes:
        MASK = (_hab_not_ignored_dtypes == 'obj')
        IDXS = ', '.join(map(str, np.array(list(_hab_not_ignored))[MASK]))
        raise ValueError(
            f"cannot use handle_as_bool on str/object columns "
            f"--- column index(es) == {IDXS}"
        )

    del _ic_hab_intersection

    return np.array(list(_hab_not_ignored))






