# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Iterable, Literal
from typing_extensions import Union

import numbers
import numpy as np

from ....utilities._nan_masking import nan_mask_numerical

from .._validation._n_features_in import _val_n_features_in
from .._validation._feature_names_in import _val_feature_names_in



def _val_ign_cols_hab_callable(
    _fxn_output: Union[Iterable[str], Iterable[numbers.Integral]],
    _first_fxn_output: Union[Iterable[str], Iterable[numbers.Integral], None],
    _name: Literal['ignore_columns', 'handle_as_bool'],
    _n_features_in: int,
    _feature_names_in: Union[Iterable[str], None]
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

    If the callable returned a vector of integers, validate the minimum
    and maximum values of the callable's returned indices are within the
    bounds of n_features_in.

    Validate the current output of the callable exactly matches the
    output from the first call to it, if applicable, as held in the
    '_first_function_output' parameter. If this is the first pass,
    '_first_function_output' must be None. If unequal, raise exception.


    Parameters
    ----------
    _fxn_output:
        Union[Iterable[str], Iterable[numbers.Integral]] - the output of
        the callable used for ignore_columns or handle_as_bool
    _first_fxn_output:
        Union[Iterable[str], Iterable[numbers.Integral], None] - the
        output of the callable on the first call to :methods: partial_fit
        or transform. used to validate that all subsequent outputs of
        the callable equal the first.
    _name:
        Literal['ignore_columns', 'handle_as_bool'] - the name of the
        parameter for which a callable was passed
    _n_features_in:
        int - the number of features in the data
    _feature_names_in:
        Union[Iterable[str], None] - the feature names of a data-bearing
        object


    Return
    ------
    -
        None


    """


    _val_n_features_in(_n_features_in)

    _val_feature_names_in(
        _feature_names_in,
        _n_features_in
    )

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
        f"or all strings indicating column names. "
    )

    # do not use the generic validation from _val_ignore_columns_handle_as_bool
    # here. use the special verbiage for callables.

    # pass the most current callable output thru this just in case the output
    # has mutated into garbage. if after the first pass, we know the first
    # pass was good, so the output would have changed significantly.

    # verify is iterable -- -- -- -- -- -- -- -- -- -- -- -- -- --
    _addon = f"got type {type(_fxn_output)}"
    try:
        iter(_fxn_output)
        if isinstance(_fxn_output, (str, dict)):
            raise Exception
        if len(np.array(list(_fxn_output)).shape) != 1:
            _addon = f"got {len(np.array(list(_fxn_output)).shape)}D."
            raise Exception
    except:
        raise TypeError(err_msg + _addon)
    del _addon

    # dont need to validate _first_fxn_output here. it would have gone
    # thru the validation on the first pass and should not have changed
    # since.
    # END verify is iterable -- -- -- -- -- -- -- -- -- -- -- -- -- --


    is_empty, is_str, is_num = False, False, False
    if len(_fxn_output) == 0:
        is_empty = True
    # verify the callable returned an iterable holding ints or strs.
    # do not use .astype(np.float64) to check if is num/str!
    # ['0787', '5927', '4060', '2473'] will pass and be treated as
    # column indices when they are actually column headers.
    elif all(map(isinstance, _fxn_output, (str for _ in _fxn_output))):
        is_str = True
    elif all(map(
        isinstance, _fxn_output, (numbers.Real for _ in _fxn_output)
    )):
        # ensure all are integers
        if any(map(isinstance, _fxn_output, (bool for _ in _fxn_output))):
            raise TypeError(err_msg + f"got a boolean.")
        if np.any(nan_mask_numerical(np.array(list(_fxn_output), dtype=object))):
            raise TypeError(err_msg + f"got a nan-like value.")
        if not all(map(lambda x: int(x)==x, _fxn_output)):
            raise TypeError(err_msg + f"got a non-integer number.")
        is_num = True
    else:
        raise TypeError(err_msg + f"got a non-string/numeric value.")
    # END if list-like validate contents are all str or all int ** * ** *

    del err_msg

    assert (is_empty + is_str + is_num) == 1

    # if on a later pass, dont need to check internals, check directly
    # against the first output.
    if _first_fxn_output is not None:
        if not np.array_equal(_fxn_output, _first_fxn_output):
            raise ValueError(
                f"every call to the '{_name}' callable must produce the same "
                f"output across a series of partial fits or transforms. the "
                f"current output is different than the first seen output. "
                f"\ngot: {_fxn_output}"
                f"\nexpected: {_first_fxn_output}"
            )

        return

    # v v v all of this should only be accessed on the first partial_fit/transform

    if is_empty:
        pass
    elif is_str:
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
    elif is_num:
        _err_msg = lambda _value: (
            f"the '{_name}' callable produced a vector of indices but "
            f"column index {_value} is out of bounds for data with "
            f"{_n_features_in} features"
        )
        if min(_fxn_output) < -_n_features_in:
            raise ValueError(_err_msg(min(_fxn_output)))
        if max(_fxn_output) >= _n_features_in:
            raise ValueError(_err_msg(max(_fxn_output)))
        del _err_msg
    else:
        raise Exception




























