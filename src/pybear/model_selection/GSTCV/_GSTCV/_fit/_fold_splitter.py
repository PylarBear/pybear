# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
from ..._type_aliases import GenericSlicerType

from .._type_aliases import (
    SKSlicerType,
    XSKWIPType,
    YSKWIPType
)

import numpy as np
import pandas as pd
import scipy.sparse as ss



def _fold_splitter(
    train_idxs: Union[GenericSlicerType, SKSlicerType],
    test_idxs: Union[GenericSlicerType, SKSlicerType],
    *data_objects: Union[XSKWIPType, YSKWIPType],
) -> tuple[tuple[XSKWIPType, YSKWIPType], ...]:

    """
    Split given data objects into train / test pairs using the given
    train and test indices. The train and test indices independently
    slice the given data objects; the entire data object need not be
    consumed in a train / test split and the splits can also possibly
    share indices. Standard indexing rules apply. Returns a tuple whose
    length is equal to the number of data objects passed, holding tuples
    of the train / test splits for the respective data objects.
    train_idxs and test_idxs must be 1D vectors of indices, not booleans.


    Parameters
    ----------
    # pizza fix these type hints!
    train_idxs:
        Iterable[int] - 1D vector of row indices used to slice train sets
        out ouf every given data object.
    test_idxs:
        Iterable[int] - 1D vector of row indices used to slice test sets
        out ouf every given data object.
    *data_objects:
        Union[XSKWIPType, YSKWIPType] - The data objects to slice.
        Need not be of equal size, and need not be completely consumed
        in the train / test splits. However, standard indexing rules
        apply when slicing by train_idxs and test_idxs.


    Return
    ------
    -
        pizza
        SPLITS: tuple[tuple[npt.NDArray, npt.NDArray], ...] - return
        the train / test splits for the given data objects in the order
        passed in a tuple of tuples, each inner tuple containing a
        train/test pair.

    """


    # helper ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    def _val_cond_idxs(_name, _idxs):
        """Helper for validating and conditioning slicers."""
        _err_msg = f"'{_name}' must be a 1D vector of integers, not bools."
        _idxs = np.array(_idxs)
        if _idxs.dtype == bool:
            raise TypeError(_err_msg)
        if len(_idxs.shape) != 1:
            raise ValueError(_err_msg)
        del _err_msg
        return _idxs
    # END helper ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    train_idxs = _val_cond_idxs('train_idxs', train_idxs)

    test_idxs = _val_cond_idxs('test_idxs', test_idxs)

    del _val_cond_idxs


    SPLITS = []
    for _data in data_objects:

        # pizza u will be coming back to here to deal with what containers are allowed.
        # if 'shape' is required then this precludes py containers
        assert len(_data.shape) in [1, 2], f"data objects must be 1-D or 2-D"

        if isinstance(_data, np.ndarray):
            _data_train = _data[train_idxs]
            _data_test = _data[test_idxs]
        elif isinstance(_data, pd.DataFrame):
            _data_train = _data.iloc[train_idxs, :]
            _data_test = _data.iloc[test_idxs, :]
        elif hasattr(_data, 'toarray'):
            _og_type = type(_data)
            _data = ss.csr_array(_data)
            _data_train = _og_type(_data[train_idxs, :])
            _data_test = _og_type(_data[test_idxs, :])
            _data = _og_type(_data)
            del _og_type
        elif hasattr(_data, 'clone'):
            _bool_train_idxs = np.zeros(_data.shape[0]).astype(bool)
            _bool_train_idxs[train_idxs] = True
            _data_train = _data.filter(_bool_train_idxs)
            del _bool_train_idxs
            _bool_test_idxs = np.zeros(_data.shape[0]).astype(bool)
            _bool_test_idxs[test_idxs] = True
            _data_test = _data.filter(_bool_test_idxs)
            del _bool_test_idxs
        else:
            _data_train = _data[train_idxs]
            _data_test = _data[test_idxs]


        SPLITS.append(tuple((_data_train, _data_test)))


    return tuple(SPLITS)







