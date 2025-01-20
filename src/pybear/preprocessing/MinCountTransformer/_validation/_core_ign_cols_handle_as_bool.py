# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing_extensions import Union
import numpy as np

from .._type_aliases import IgnColsHandleAsBoolDtype
from ._val_n_features_in import _val_n_features_in


def _core_ign_cols_handle_as_bool(
    _kwarg_value: IgnColsHandleAsBoolDtype,
    _name: str,
    _mct_has_been_fit: bool=False,  # pass hasattr(self, 'n_features_in_')
    _n_features_in: Union[None, int]=None,
    _feature_names_in: Union[None, np.ndarray[str]]=None
) -> Union[callable, np.ndarray[int], np.ndarray[str]]:

    """
    Validate containers and content for shared attributes of ignore_columns and
    handle_as_bool.

    Validate:
    - container is iterable, callable, or None
    - if iterable, contains valid strings or integers

    Parameters
    ----------
    _kwarg_value:
        Union[Iterable[str] Iterable[int], callable, None] - The current
            ignore_columns or handle_as_bool object.

    _name:
        str - Name of the object.

    Returns
    ----------
    -
        _kwarg_value:
            Union[np.ndarray[int], np.ndarray[str], callable] - Standardized container.

    """

    if _name.lower() not in ['handle_as_bool', 'ignore_columns']:
        raise ValueError(f"'_name' must be 'handle_as_bool' or 'ignore_columns'")

    if not isinstance(_mct_has_been_fit, bool):
        raise TypeError(f"'_mct_has_been_fit must be boolean")

    if _mct_has_been_fit:
        if _n_features_in is None:
            raise ValueError(f'if _mct_has_been_fit is True, must also pass '
                             f'_n_features_in')
        elif _n_features_in is not None:
            _n_features_in = _val_n_features_in(_n_features_in)

            if _feature_names_in is not None:
                if not isinstance(_feature_names_in, np.ndarray):
                    raise TypeError(
                        f"_feature_names_in must be a numpy ndarray of strings"
                    )

                _ = len(_feature_names_in)
                __ = _n_features_in
                if _ != __:
                    raise ValueError(f"len(_feature_names_in) ({_}) must equal "
                                     f"n_features_in ({__})")
                del _, __

    elif not _mct_has_been_fit:
        if _n_features_in is not None:
            raise ValueError(f"not allowed to pass _n_features in if "
                             f"_mct_has_been_fit is False")
        if _feature_names_in is not None:
            raise ValueError(f"not allowed to pass _feature_names_in in if "
                             f"_mct_has_been_fit is False")

    while True:
        # ESCAPE VALIDATION IF _kwarg_value IS CALLABLE AND JUST RETURN THE CALLABLE

        # validate outer container / escape if is callable ** * ** * ** * ** *

        err_msg = (f"{_name} must be None, a list-like, or a callable that "
                   f"returns a list-like")

        if _kwarg_value is None:
            _kwarg_value = np.array([], dtype=np.uint32)
            break
        elif callable(_kwarg_value):
            break

        try:
            iter(_kwarg_value)
            if isinstance(_kwarg_value, (str, dict)):
                raise Exception
        except:
            raise TypeError(err_msg)

        del err_msg
        # END validate outer container / escape if is callable ** * ** * ** * **

        # if list-like validate contents are all str or all int ** * ** * ** *
        err_msg = (f"if {_name} is passed as a list-like, it must contain all "
            f"integers indicating column indices or all strings indicating "
            f"column names")

        try:
            _kwarg_value = np.array(sorted(list(set(_kwarg_value))), dtype=object)
        except:
            raise TypeError(err_msg) from None

        _dtypes = []
        # if _kwarg_value was None it was converted to [] and skips this
        for idx, value in enumerate(_kwarg_value):
            if isinstance(value, str):
                _dtypes.append('str')
                continue

            try:
                float(value)
                if isinstance(value, bool):
                    raise Exception
                if int(value) != value:
                    raise Exception
                _kwarg_value[idx] = int(value)
                _dtypes.append('int')
            except:
                raise TypeError(err_msg)

        if len(_dtypes) != len(_kwarg_value):
            raise Exception(f"Error building _dtypes, len != len({_name})")

        if len(np.unique(_dtypes)) not in [0, 1]:
            raise ValueError(err_msg)

        try:
            _dtype = _dtypes[0]
        except:
            pass

        del err_msg, _dtypes

        # END if list-like validate contents are all str or all int ** * ** *

        # if MCT has been fit, validate list-like against characteristics of X
        if _mct_has_been_fit:

            # _feature_names_in is not necessarily available, could be None

            # _kwarg_value could be an empty list, and this would be skipped
            for idx, _column in enumerate(_kwarg_value):
                if _dtype == 'int':
                    if max(_kwarg_value) >= _n_features_in:
                        raise ValueError(f"{_name} index {max(_kwarg_value)} is out "
                            f"of bounds for data with {_n_features_in} columns")
                    break  # only look inside _kwarg_value once if _dtype is int
                    # keep this under the for loop, to keep the for loop
                    # controlling entry to 'int' and 'str' logic
                elif _dtype == 'str':
                    if _feature_names_in is not None:
                        if _column not in _feature_names_in:
                            raise ValueError(f'{_name} entry, column "{_column}", '
                                f'is not in column names seen during original fit')
                        # CONVERT COLUMN NAMES TO COLUMN INDEX
                        MASK = (_feature_names_in == _column)
                        _kwarg_value[idx] = np.arange(_n_features_in)[MASK][0]
                        del MASK
                    else: # feature_names_in_ is None
                        raise ValueError(f"{_name} as list-like can only contain indices "
                            f"when the data is not passed with column names")

            _kwarg_value = _kwarg_value.astype(np.uint32)

        # END if MCT has been fit, validate list-like against characteristics of X

        break


    return _kwarg_value
























