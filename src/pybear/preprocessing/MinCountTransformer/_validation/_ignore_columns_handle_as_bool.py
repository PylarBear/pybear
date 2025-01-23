# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Literal
from typing_extensions import Union
from .._type_aliases import IgnoreColumnsType, HandleAsBoolType
import numpy.typing as npt

import numbers
import numpy as np

from ....utilities._nan_masking import nan_mask_numerical



def _val_ignore_columns_handle_as_bool(
    _value: Union[IgnoreColumnsType, HandleAsBoolType],
    _name: Literal['ignore_columns', 'handle_as_bool'],
    _n_features_in: int,
    _feature_names_in: Union[npt.NDArray[str], None]=None
) -> None:

    """
    Validate ignore_columns or handle_as_bool.

    Validate:
    - container is iterable, callable, or None
    - if iterable, contains integers or valid strings

    - name is either 'ignore_columns' or 'handle_as_bool' only

    - _n_features_in is passed and is int >= 0

    - _feature_names_in is NDArray[str] or None


    Parameters
    ----------
    _value:
        Union[Iterable[str], Iterable[int], callable, None] - the value
        passed for the 'ignore_columns' or 'handle_as_bool' parameter to
        the MinCountTransformer instance.


    Return
    ------
    -
        None

    """

    # other validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    assert isinstance(_n_features_in, int)
    assert not isinstance(_n_features_in, bool)
    assert _n_features_in >= 0

    assert isinstance(_name, str)
    _name = _name.lower()
    assert _name in ['ignore_columns', 'handle_as_bool']

    if _feature_names_in is not None:
        assert isinstance(_feature_names_in, np.ndarray)
        assert len(_feature_names_in) == _n_features_in
        assert all(map(
            isinstance, _feature_names_in, (str for _ in _feature_names_in)
        ))
    # END other validation ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # ESCAPE VALIDATION IF _value IS CALLABLE OR None

    if _value is None:
        return

    elif callable(_value):
        return


    err_msg = (f"'{_name}' must be None, a list-like, or a callable "
               f"that returns a list-like")

    try:
        iter(_value)
        if isinstance(_value, (str, dict)):
            raise Exception
    except:
        raise TypeError(err_msg)

    del err_msg


    # if list-like, validate contents are all str or all int ** * ** * ** *

    err_msg = (f"if '{_name}' is passed as a list-like, it must contain "
        f"all integers indicating column indices or all strings indicating "
        f"column names"
    )

    # do not use .astype(np.float64) to check if is num/str!
    # ['0787', '5927', '4060', '2473'] will pass and be treated as
    # column indices when they are actually column headers.
    if len(_value) == 0:
        is_int, is_str, is_empty = False, False, True
    elif all(map(
        isinstance, _value, (str for _ in _value)
    )):
        is_int, is_str, is_empty = False, True, False
    elif all(map(
        isinstance, _value, (numbers.Real for _ in _value)
    )):
        # ensure all are integers
        if any(map(
            isinstance, _value, (bool for _ in _value)
        )):
            raise TypeError(err_msg)
        if np.any(nan_mask_numerical(np.array(_value, dtype=object))):
            raise TypeError
        if not all(map(lambda x: int(x)==x, _value)):
            raise TypeError(err_msg)
        _value = sorted(list(map(int, _value)))
        is_int, is_str, is_empty = True, False, False
    else:
        raise TypeError(err_msg)
    # END if list-like validate contents are all str or all int ** * ** *

    if len(set(_value)) != len(_value):
        raise ValueError(f"there are duplicate values in {_name}")

    # validate list-like against characteristics of X ** ** ** ** ** ** **

    # _feature_names_in is not necessarily available, could be None

    if is_empty:
        pass
    elif is_int:
        if any(map(lambda x: x < -_n_features_in, _value)):
            raise ValueError(
                f"'{_name}' index {min(_value)} is out of bounds for "
                f"data with {_n_features_in} columns"
            )

        if any(map(lambda x: x >= _n_features_in, _value)):
            raise ValueError(
                f"'{_name}' index {max(_value)} is out of bounds for "
                f"data with {_n_features_in} columns"
            )
    elif is_str:
        if _feature_names_in is not None:
            # if _value is empty list-like, this is bypassed
            for _column in _value:
                if _column not in _feature_names_in:
                    raise ValueError(
                        f"'{_name}' entry column '{_column}', is not in "
                        f"the passed column names"
                    )
        else: # feature_names_in_ is None
            raise ValueError(
                f"when the data is passed without column names '{_name}' "
                f"as list-like can only contain indices"
            )
    else:
        raise Exception

    del is_int, is_str, is_empty

    # END validate list-like against characteristics of X ** ** ** ** ** **










