# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from .._type_aliases import DataType, InstructionType

import numpy as np
import pandas as pd
import scipy.sparse as ss




def _transform(
    _X: DataType,
    _instructions: InstructionType
) -> DataType:


    """
    Talking pizza?



    Parameters
    ----------
    _X: DataType,
    _instructions: InstructionType


    Return
    ------
    -
        _X:


    """

    # class InstructionType(TypedDict):
    #
    #     keep: Required[Union[None, list, npt.NDArray[int]]]
    #     delete: Required[Union[None, list, npt.NDArray[int]]]
    #     add: Required[Union[None, dict[str, any]]]

    # 'keep' isnt needed to modify X, it is only in the dictionary for
    # ease of making self.kept_columns_ later.

    KEEP_MASK = np.ones(_X.shape[1]).astype(bool)
    KEEP_MASK[_instructions['delete']] = False

    if isinstance(_X, np.ndarray):
        _X = _X[:, KEEP_MASK]
        if _instructions['add']:
            _key = list(_instructions['add'].keys())[0]
            _X = np.hstack((_X, _instructions['add'][_key]))
            del _key

    elif isinstance(_X, pd.core.frame.DataFrame):
        _X = _X.loc[:, KEEP_MASK]
        if _instructions['add']:
            _key = list(_instructions['add'].keys())[0]
            _X[_key] = _instructions['add'][_key]
            del _key

    elif hasattr(_X, 'toarray'):     # scipy.sparse
        _X = _X[:, KEEP_MASK]
        if _instructions['add']:
            _key = list(_instructions['add'].keys())[0]
            _X = ss.hstack((_X, _instructions['add'][_key]))
            del _key

    else:
        raise TypeError(f"Unknown dtype {type(_X)} in transform().")


    return _X











