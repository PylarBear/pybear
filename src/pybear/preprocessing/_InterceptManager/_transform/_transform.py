# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import InstructionType
from typing_extensions import Union
import numpy.typing as npt

import numpy as np
import pandas as pd
import scipy.sparse as ss




def _transform(
    _X: Union[npt.NDArray, pd.DataFrame, ss.csc_array, ss.csc_matrix],
    _instructions: InstructionType
) -> Union[npt.NDArray, pd.DataFrame, ss.csc_array, ss.csc_matrix]:


    """
    Manage the constant columns in X. Apply the removal criteria given
    by :param: keep via _instructions to the constant columns found
    during fit.


    Parameters
    ----------
    _X:
        {array-like, scipy sparse csc} of shape (n_samples, n_features) -
        The data to be transformed. Must be numpy ndarray, pandas
        dataframe, or scipy sparse csc matrix/array.

    _instructions:
        TypedDict[
            keep: Required[Union[None, list, npt.NDArray[int]]],
            delete: Required[Union[None, list, npt.NDArray[int]]],
            add: Required[Union[None, dict[str, any]]]
        ] - instructions for keeping, deleting, or adding constant
        columns.


    Return
    ------
    -
        X: {array-like, scipy sparse csc} of shape (n_samples,
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
            # whatever dtype _X is (with some caveats)

            # str dtypes are changing here. also on windows int dtypes
            # are changing to int64.
            _X = np.hstack((
                _X,
                np.full((_X.shape[0], 1), _value)
            ))
            # there does not seem to be an obvious connection between what
            # the dtype of _value is and the resultant dtype (for example,
            # _X with dtype '<U10' when appending float(1.0), the output dtype
            # is '<U21' (???, maybe floating point error on the float?) )

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

    elif isinstance(_X, (ss.csc_matrix, ss.csc_array)):
        # remove the columns
        _X = _X[:, KEEP_MASK]
        # if :param: keep is dict, add the new intercept
        if _instructions['add']:
            _key = list(_instructions['add'].keys())[0]
            _value = _instructions['add'][_key]
            _X = ss.hstack((
                _X,
                type(_X)(np.full((_X.shape[0], 1), _value))
            ))
            del _key

    else:
        raise TypeError(f"Unknown dtype {type(_X)} in _transform().")


    return _X











