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
    Manage the constant columns in X. Apply the removal criteria given
    by :param: keep via _instructions to the constant columns found
    during fit.


    Parameters
    ----------
    _X:
        {array-like, scipy sparse matrix} of shape (n_samples,
        n_features) - The data to be transformed.
    _instructions:
        dict[Literal['keep']: Union[list[int], None],
            Literal['delete']: Union[list[int], None].
            Literal['add']: Union[dict[str, any], None]] -
        instructions for keeping, deleting, or adding constant columns.


    Return
    ------
    -
        X: {array-like, scipy sparse matrix} of shape (n_samples,
            n_transformed_features) - The transformed data.


    """

    # class InstructionType(TypedDict):
    #
    #     keep: Required[Union[None, list, npt.NDArray[int]]]
    #     delete: Required[Union[None, list, npt.NDArray[int]]]
    #     add: Required[Union[None, dict[str, any]]]

    # 'keep' isnt needed to modify X, it is only in the dictionary for
    # ease of making self.kept_columns_ later.

    # build the mask that will take out deleted columns
    KEEP_MASK = np.ones(_X.shape[1]).astype(bool)
    if _instructions['delete'] is not None:
        # if _instructions['delete'] is None numpy actually maps
        # assignment to all positions! so that means we must put this
        # statement under an if that only allows when not None
        KEEP_MASK[_instructions['delete']] = False


    if isinstance(_X, np.ndarray):

        # remove the columns
        _X = _X[:, KEEP_MASK]
        # if :param: keep is dict, add the new intercept

        if _instructions['add']:
            _key = list(_instructions['add'].keys())[0]
            _value = _instructions['add'][_key]
            # this just rams the fill value into _X, and conforms to
            # whatever dtype _X is

            _X = np.hstack((
                _X,
                np.full((_X.shape[0], 1), _value)
            ))

            del _key, _value

    elif isinstance(_X, pd.core.frame.DataFrame):
        # remove the columns
        _X = _X.iloc[:, KEEP_MASK]
        # if :param: keep is dict, add the new intercept
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
        _og_type = type(_X)  # keep this to convert back after going to csc
        # remove the columns
        _X = _X.tocsc()[:, KEEP_MASK]   # must use tocsc, COO cannot be sliced
        # if :param: keep is dict, add the new intercept
        if _instructions['add']:
            _key = list(_instructions['add'].keys())[0]
            _value = _instructions['add'][_key]
            _X = ss.hstack((
                _X,
                ss.csc_array(np.full((_X.shape[0], 1), _value))
            ))
            del _key
        _X = _og_type(_X)
        del _og_type

    else:
        raise TypeError(f"Unknown dtype {type(_X)} in transform().")


    return _X











