# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from .._type_aliases import DataFormatType, InstructionType

import numpy as np
import pandas as pd
import scipy.sparse as ss




def _transform(
    _X: DataFormatType,
    _instructions: InstructionType
) -> DataFormatType:


    """
    Talking pizza?



    Parameters
    ----------
    _X: DataFormatType,
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
    if _instructions['delete'] is not None:
        # if _instructions['delete'] is None numpy actually maps
        # assignment to all positions!
        KEEP_MASK[_instructions['delete']] = False


    if isinstance(_X, np.ndarray):
        _X = _X[:, KEEP_MASK]
        if _instructions['add']:
            _key = list(_instructions['add'].keys())[0]
            _X = np.hstack((_X, np.full((_X.shape[0], 1), _instructions['add'][_key])))
            del _key

    elif isinstance(_X, pd.core.frame.DataFrame):
        _X = _X.loc[:, KEEP_MASK]
        if _instructions['add']:
            _key = list(_instructions['add'].keys())[0]
            _value = _instructions['add'][_key]
            try:
                float(_value)
                _is_num = True
            except:
                _is_num = False

            _dtype = np.float64 if _is_num else object

            _X[_key] = np.full((_X.shape[0],), _value).astype(_dtype)

            del _key, _value, _is_num, _dtype

    elif hasattr(_X, 'toarray'):     # scipy.sparse
        _og_type = type(_X)
        _X = _X.tocsc()[:, KEEP_MASK]   # must use tocsc, COO cannot be sliced
        if _instructions['add']:
            _key = list(_instructions['add'].keys())[0]
            _X = ss.hstack((
                _X,
                type(_X)(np.full((_X.shape[0], 1), _instructions['add'][_key]))
            ))
            del _key
        _X = _og_type(_X)
        del _og_type
    else:
        raise TypeError(f"Unknown dtype {type(_X)} in transform().")


    return _X











