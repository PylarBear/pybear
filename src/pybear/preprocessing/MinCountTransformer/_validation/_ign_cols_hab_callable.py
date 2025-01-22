# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Iterable, Literal
from typing_extensions import Union
import numpy.typing as npt

import numbers
import numpy as np

from ....utilities._nan_masking import nan_mask_numerical



def _val_ign_cols_hab_callable(
    _fxn_output: Union[Iterable[str], Iterable[numbers.Integral]],
    _name: Literal['ignore_columns', 'handle_as_bool'],
    _feature_names_in: Union[npt.NDArray[str], None]
) -> None:

    """
    Validate a callable used for ignore_columns or handle_as_bool returns
    either:

    - a 1D list-like full of integers
    - a 1D list-like full of strings
    - an empty 1D list-like

    If the callable returned a vector of strings and feature_names_in is
    provided, validate the strings are in feature_names_in. If
    feature_names_in is not provided, raise exception, feature names
    cannot be mapped to column indices if there are no feature names.

    If the callable returned a vector of integers and feature_names_in
    is provided, validate the minimum and maximum values of the
    callable's returned indices are within the dimensions of
    feature_names_in. If feature_names_in is not provided, skip
    validation.


    Parameters
    ----------
    _fxn_output:
        Iterable[numbers.Integral, str] - the output of the callable used
        for ignore_columns or handle_as_bool
    _name:
        Literal['ignore_columns', 'handle_as_bool'] - the name of the
        parameter for which a callable was passed
    _feature_names_in:
        Union[NDArray[str], None] - the feature names of a data-bearing
        object


    Return
    ------
    -
        None


    """


    # validate feature_names_in ** * ** * ** * ** * ** * ** * ** * ** * ** *
    err_msg = (
        f"'feature_names_in' must be None or a 1D vector of strings "
        f"indicating the feature names of a data-bearing object"
    )
    try:
        if _feature_names_in is None:
            raise UnicodeError
        iter(_feature_names_in)
        if isinstance(_feature_names_in, (str, dict)):
            raise Exception
        if not all(map(
            isinstance, _feature_names_in, (str for _ in _feature_names_in)
        )):
            raise MemoryError
    except UnicodeError:
        pass
    except MemoryError:
        raise ValueError(err_msg)
    except:
        raise TypeError(err_msg)

    del err_msg
    # END validate feature_names_in ** * ** * ** * ** * ** * ** * ** * ** *

    # validate _name ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    err_msg = f"'_name' must be 'ignore_columns' or 'handle_as_bool'"
    if not isinstance(_name, str):
        raise TypeError(err_msg)
    _name = _name.lower()
    if _name not in ['ignore_columns', 'handle_as_bool']:
        raise ValueError(err_msg)
    del err_msg
    # END validate _name ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    err_msg = (
        f"{_name}: when a callable is used, the callable must return a "
        f"1D list-like containing all integers indicating column indices "
        f"or all strings indicating column names"
    )

    # verify is iterable -- -- -- -- -- -- -- -- -- -- -- -- -- --
    try:
        iter(_fxn_output)
        if isinstance(_fxn_output, (str, dict)):
            raise Exception
        _fxn_output = np.array(list(_fxn_output), dtype=object)
        if len(_fxn_output.shape) != 1:
            raise Exception
    except:
        raise TypeError(err_msg)
    # END verify is iterable -- -- -- -- -- -- -- -- -- -- -- -- -- --

    if len(_fxn_output) == 0:
        return

    # verify the callable returned an iterable holding ints or strs.
    # do not use .astype(np.float64) to check if is num/str!
    # ['0787', '5927', '4060', '2473'] will pass and be treated as
    # column indices when they are actually column headers.
    is_str = False
    if all(map(isinstance, _fxn_output, (str for _ in _fxn_output))):
        is_str = True
    elif all(map(
        isinstance, _fxn_output, (numbers.Real for _ in _fxn_output)
    )):
        # ensure all are integers
        if any(map(isinstance, _fxn_output, (bool for _ in _fxn_output))):
            raise TypeError(err_msg)
        if np.any(nan_mask_numerical(np.array(_fxn_output))):
            raise TypeError(err_msg)
        if not all(map(lambda x: int(x)==x, _fxn_output)):
            raise TypeError(err_msg)
    else:
        raise TypeError(err_msg)
    # END if list-like validate contents are all str or all int ** * ** *

    del err_msg

    if is_str:
        if _feature_names_in is None:
            raise ValueError(
                f"the '{_name}' callable produced a vector of strings but "
                f"the features names of the data are not provided. if feature "
                f"names are not available, then the callable must produce a "
                f"vector of integers."
            )
        elif _feature_names_in is not None:   # must be 1D vector of strings

            for _feature in _fxn_output:
                if _feature not in _feature_names_in:
                    raise ValueError(
                        f"the feature name '{_feature}' produced by the "
                        f"{_name} callable is not in 'feature_names_in'"
                    )

    elif not is_str:
        if _feature_names_in is None:
            # without knowing feature_names_in, all we have is a _fxn_output
            # vector with integers that may be all positive, all negative, or a
            # mix of both, and we cant validate the values. so just skip.
            pass

        elif _feature_names_in is not None:  # must be 1D vector of strings

            _n_features_in = len(_feature_names_in)
            if min(_fxn_output) < -_n_features_in:
                raise ValueError(
                    f"column index {min(_fxn_output)} is out of bounds for a "
                    f"feature name vector with {_n_features_in} features"
                )
            if max(_fxn_output) >= _n_features_in:
                raise ValueError(
                    f"column index {max(_fxn_output)} is out of bounds for a "
                    f"feature name vector with {_n_features_in} features"
                )




























