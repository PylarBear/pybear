# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from typing import Iterable
from typing_extensions import Union
import numpy as np




def _val_ign_cols_hab_callable(
        _ign_cols_hab_value: Union[Iterable[str], Iterable[int]],
        _name: str
    ) -> None:

    """
    Validate a callable used for ignore_columns or handle_as_bool returns
    either
    - a list-like full of integers >= 0
    - a list-like full of strings (verify strings against column names
        is elsewhere)

    Parameters
    ----------
    _ign_cols_hab_value: Any - the output of the callable used for
        ignore_columns or handle_as_bool
    _name: str - 'ignore_columns' or 'handle_as_bool'

    Return
    ------
    -
        None

    """



    err_msg = (f"{_name}: when a callable is used, the callable must return a "
               f"list-like containing all integers >= 0 indicating column "
               f"indices or all strings indicating column names")

    # verify the callable returned an iterable holding ints or strs
    try:
        iter(_ign_cols_hab_value)
        if isinstance(_ign_cols_hab_value, (dict, str)):
            raise Exception
        _ign_cols_hab_value = list(_ign_cols_hab_value)
    except:
        raise TypeError(err_msg)


    try:
        _ign_cols_hab_value = list(map(float, _ign_cols_hab_value))

        np_ichv = np.array(_ign_cols_hab_value)
        assert np.array_equiv(np_ichv.astype(np.int32), np_ichv)

    except AssertionError:
        raise TypeError(err_msg)

    except:
        _ign_cols_hab_value = list(map(str, _ign_cols_hab_value))


































