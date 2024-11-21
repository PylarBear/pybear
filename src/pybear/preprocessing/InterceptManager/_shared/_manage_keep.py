# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import KeepType, DataFormatType
from typing import Literal
from typing_extensions import Union
import numpy.typing as npt
import numpy as np
import warnings



def _manage_keep(
    _keep: KeepType,
    _X: DataFormatType,
    _constant_columns: dict[int, any],
    _n_features_in: int,
    _feature_names_in: Union[npt.NDArray[str], None]
) -> Union[Literal['first', 'last', 'random', 'none'], dict[str, any], int]:

    """

    FROM columns & keep VALIDATION WE KNOW:
    'feature_names_in_' is either type(None) or Iterable[str] whose len == X.shape[1]

    if 'keep' is dict:
    	len == 1
    	key is str
    	warns if 'feature_names_in_' is not None and key is in 'feature_names_in_'
    if 'keep' is number:
    	is int
    	is not bool
    	is in range(X.shape[1])
    if 'keep' is callable(X):
        pizza revisit this after us decide about _keep_and_columns
    	output is int
    	output is not bool
    	output is in range(X.shape[1])
    if 'keep' is str in ('first', 'last', 'random', 'none'):
    	if 'feature_names_in_' is not None, keep literal is not in 'feature_names_in_'
    if 'keep' is str not in literals:
    	'feature_names_in_' is cannot be None
    	'keep' must be in 'feature_names_in_'

    'feature_names_in_' could be None or Iterable[str]
    'keep' could be dict[str, any], int, callable(X), 'first', 'last', 'random', 'none', a feature name

    'feature_names_in_' is not changed
    'keep':
          dict[str, any]                       ---- passes thru, not validated
          callable(X)                          ---- converted to int, validated
          'first', 'last', 'random', 'none'    ---- converted to int, validated
          a feature name                       ---- converted to int, validated
          int                                  ---- validated


    Parameters
    ----------


    Return
    ------



    """



    if isinstance(_keep, dict):
        __keep = _keep
    elif callable(_keep):
        __keep = _keep(_X)
        if __keep not in _constant_columns:
            raise ValueError(
                f"'keep' callable has returned an integer column index ({_keep}) "
                f"that is not a column of constants. \nconstant columns: "
                f"{_constant_columns}"
            )
    elif isinstance(_keep, str) and _feature_names_in is not None and _keep in _feature_names_in:
        # if keep is str, convert to idx
        # validity of keep as str (header was passed, keep is in header)
        # should have been validated in _validation > _keep_and_columns
        # dont need to validate str keep w header is None, was done in validation
        __keep = int(np.arange(_n_features_in)[_feature_names_in == _keep][0])
        # this is the first place where we could validate whether the _keep str is actually
        # a constant column in the data
        if __keep not in _constant_columns:
            raise ValueError(
                f"'keep' was passed as '{_keep}' corresponding to column index ({_keep}) "
                f"which is not a column of constants. \nconstant columns: "
                f"{_constant_columns}"
            )
    elif _keep in ('first', 'last', 'random', 'none'):
        _sorted_constant_column_idxs = sorted(list(_constant_columns))
        if len(_sorted_constant_column_idxs) == 0:
            warnings.warn(f"ignoring :param: keep '{_keep}', there are no constant columns")
            __keep = 'none'
        elif _keep == 'first':
            __keep = int(_sorted_constant_column_idxs[0])
        elif _keep == 'last':
            __keep = int(_sorted_constant_column_idxs[-1])
        elif _keep == 'random':
            __keep = int(np.random.choice(_sorted_constant_column_idxs))
        elif _keep == 'none':
            __keep = 'none'
    elif isinstance(_keep, int):
        # this is the first place where we could validate whether the _keep int is actually
        # a constant column in the data
        __keep = _keep
        if __keep not in _constant_columns:
            raise ValueError(
                f"'keep' was passed as column index ({_keep}) "
                f"which is not a column of constants. \nconstant columns: "
                f"{_constant_columns}"
            )
    else:
        raise AssertionError(f"algorithm failure. invalid 'keep': {_keep}")


    # __keep could be dict[str, any], int, or 'none'
    return __keep












