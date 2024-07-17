# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from typing import Union

import numpy as np
import numpy.typing as npt
import pandas as pd

from model_selection.GSTCV._type_aliases import (
    DataType,
    GenericKFoldType,
    SKKFoldType,
    XSKWIPType,
    YSKWIPType
)


def _fold_splitter(
        train_idxs: Union[GenericKFoldType, SKKFoldType],
        test_idxs: Union[GenericKFoldType, SKKFoldType],
        *data_objects: Union[XSKWIPType, YSKWIPType],
    ) -> tuple[tuple[XSKWIPType, YSKWIPType]]:


    """
    Split given data objects into train / test pairs using the given
    train and test indices. The train and test indices independently
    slice the given data objects; the entire data object need not be
    consumed in a train / test split and the splits can also possibly
    share indices. Standard indexing rules apply. Returns a tuple whose
    length is equal to the number of data objects passed, holding tuples
    of the train / test splits for the respective data objects. The data
    objects must be numpy 2D arrays or 1D vectors. train_idxs and
    test_idxs must be vectors of indices, not booleans.

    Parameters
    ----------
    train_idxs:
        Iterable[int] - 1D vector of row indices used to slice train sets
        out ouf every given data object.
    test_idxs:
        Iterable[int] - 1D vector of row indices used to slice test sets
        out ouf every given data object.
    *data_objects:
        Union[XSKWIPType, YSKWIPType] - Numpy 2D arrays or 1D vectors.
        Need not be of equal size, and need not be completely consumed
        in the train test splits. However, standard indexing rules apply
        when slicing by train_idxs and test_idxs.

    Return
    ------
    -
        SPLITS:
            tuple[tuple[npt.NDArray[DataType], npt.NDArray[DataType]]] -
            return the train / test splits for the given data objects in
            the order passed in a tuple of tuples, each inner tuple
            containing a train/test pair.

    """

    train_idxs = np.array(train_idxs)
    if train_idxs.dtype == bool:
        raise TypeError(
            f"'train_idxs' must be a vector of integer slicers, not bools"
        )

    if len(train_idxs.shape) != 1:
        raise ValueError(f"'train_idxs' must be a 1D vector")

    test_idxs = np.array(test_idxs)
    if test_idxs.dtype == bool:
        raise TypeError(
            f"'test_idxs' must be a vector of integer slicers, not bools"
        )

    if len(test_idxs.shape) != 1:
        raise ValueError(f"'test_idxs' must be a 1D vector")


    SPLITS = []
    for _data in data_objects:

        if isinstance(_data, np.ndarray):
            pass
        elif isinstance(_data, pd.core.series.Series):
            raise TypeError(f"A pandas series is in SK _fold_splitter, should "
                f"only be numpy array")
        elif isinstance(_data, pd.core.frame.DataFrame):
            raise TypeError(f"A pandas dataframe is in SK _fold_splitter, should "
                f"only be numpy array")
        else:
            raise TypeError(f"object of disallowed dtype '{type(_data)}' is in "
                f"SK fold_splitter()")



        assert len(_data.shape) in [1, 2], f"data objects must be 1-D or 2-D"

        # validate data_objects rows == sum of folds' rows ** * ** * ** * ** *
        # removed this 24_07_13 to allow for splits that do not use the entire
        # dataset
        # assert (len(train_idxs) + len(test_idxs)) == _data.shape[0], \
        #     "SK fold_splitter(): (len(train_idxs) + len(test_idxs)) != _data.shape[0]"
        # END validate data_objects rows == sum of folds' rows ** * ** * ** * *

        _data_train, _data_test = _data[train_idxs], _data[test_idxs]




        train_shape = _data_train.shape
        test_shape = _data_test.shape

        assert train_shape[0] == len(train_idxs), \
            "fold_splitter(): _data_train.shape[0] != len(train_idxs)"
        assert test_shape[0] == len(test_idxs), \
            "fold_splitter(): _data_test.shape[0] != len(test_idxs)"
        if len(train_shape) == 2:
            assert train_shape[1] == _data.shape[1], \
                "fold_splitter(): _data_train.shape[1] != _data.shape[1]"
            assert test_shape[1] == _data.shape[1], \
                "fold_splitter(): _data_test.shape[1] != _data.shape[1]"

        del train_shape, test_shape

        SPLITS += tuple((_data_train, _data_test))


    assert len(SPLITS) == 2 * len(data_objects), \
        "fold_splitter(): len(SPLITS) != 2*len(data_objects)"


    return tuple(SPLITS)







