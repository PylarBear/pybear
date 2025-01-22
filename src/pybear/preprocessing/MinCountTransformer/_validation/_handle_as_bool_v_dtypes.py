# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Iterable
from typing_extensions import Union
from .._type_aliases import OriginalDtypesDtype

import numbers
import numpy as np



def _val_handle_as_bool_v_dtypes(
    _handle_as_bool: Union[Iterable[numbers.Integral], None],
    _original_dtypes: OriginalDtypesDtype
) -> None:

    """
    Validate that the columns to be handled as boolean are numeric
    columns (MCT internal dtypes 'bin_int', 'int', 'float'). MCT internal
    dtype 'obj' columns cannot be handled as boolean.

    '_handle_as_bool' must be received as a 1D list-like of integers or
    None. If a list-like, this means 2 things must have happened before
    calling this function:

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
    _original_dtypes:
        npt.NDArray[Union[Literal['bin_int', 'int', 'float', 'obj']]] -
        The datatypes for each column in the dataset as determined by
        MCT. Values can be 'bin_int', 'int', 'float', or 'obj'.


    Return
    ------
    -
        None

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
        _handle_as_bool = np.array(list(_handle_as_bool))
        if not len(_handle_as_bool.shape) == 1:
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

    # original_dtypes -- -- -- -- -- -- -- -- -- --
    _allowed = ['bin_int', 'int', 'float', 'obj']
    _err_msg = (
        f"'_original_dtypes' must be a 1D vector of values in "
        f"{', '.join(_allowed)}."
    )
    try:
        iter(_original_dtypes)
        if isinstance(_original_dtypes, (str, dict)):
            raise Exception
        _original_dtypes = np.array(_original_dtypes)
        if not len(_handle_as_bool.shape) == 1:
            raise Exception
        if not all(map(
            isinstance, _original_dtypes, (str for _ in _original_dtypes)
        )):
            raise Exception
        _original_dtypes = list(map(str.lower, _original_dtypes))
        for _ in _original_dtypes:
            if _ not in ['bin_int', 'int', 'float', 'obj']:
                _addon = f"got '{_}'"
                raise UnicodeError
    except UnicodeError:
        raise ValueError(_err_msg + _addon)
    except:
        raise TypeError(_err_msg)
    # END original_dtypes -- -- -- -- -- -- -- -- -- --

    if len(_handle_as_bool) == 0:
        return

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
    # END joint -- -- -- -- -- -- -- -- -- -- -- -- -- --

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # handle_as_bool should have been converted to indices already

    __ = np.array(_original_dtypes)[_handle_as_bool]
    if 'obj' in __:
        MASK = (__ == 'obj')
        IDXS = ', '.join(map(str, np.array(_handle_as_bool)[MASK]))
        raise ValueError(
            f"cannot use handle_as_bool on str/object columns "
            f"--- column index(es) == {IDXS}"
        )






