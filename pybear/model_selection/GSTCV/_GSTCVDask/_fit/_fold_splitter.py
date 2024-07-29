# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing import Union

from dask import compute
import dask.array as da
import dask.dataframe as ddf

from model_selection.GSTCV._type_aliases import (
    GenericSlicerType,
    DaskSlicerType,
    XDaskWIPType,
    YDaskWIPType,
    SchedulerType
)


def _fold_splitter(
        train_idxs: Union[GenericSlicerType, DaskSlicerType],
        test_idxs: Union[GenericSlicerType, DaskSlicerType],
        *data_objects: Union[XDaskWIPType, YDaskWIPType],
        scheduler: SchedulerType=None
    ) -> tuple[tuple[XDaskWIPType, YDaskWIPType, ...]]:

    """
    Split given data objects into train / test pairs using the given
    train and test indices. The train and test indices independently
    slice the given data objects; the entire data object need not be
    consumed in a train / test split and the splits can also possibly
    share indices. Standard indexing rules apply. Returns a tuple whose
    length is equal to the number of data objects passed, holding tuples
    of the train / test splits for the respective data objects. The data
    objects must be dask.array.core.Array 2D arrays or 1D vectors. train_idxs
    and test_idxs must be vectors of indices, not booleans.

    Parameters
    ----------
    train_idxs:
        Iterable[int] - 1D vector of row indices used to slice train sets
        out ouf every given data object.
    test_idxs:
        Iterable[int] - 1D vector of row indices used to slice test sets
        out ouf every given data object.
    *data_objects:
        Union[XDaskWIPType, YDaskWIPType] - dask.array.core.Array 2D arrays or
        1D vectors. Need not be of equal size, and need not be completely
        consumed in the train test splits. However, standard indexing
        rules apply when slicing by train_idxs and test_idxs.

    Return
    ------
    -
        SPLITS:
            tuple[tuple[da.core.Array[float], da.core.Array[float]]] -
            return the train / test splits for the given data objects in
            the order passed in a tuple of tuples, each inner tuple
            containing a train/test pair.

    """

    train_idxs = da.array(train_idxs)
    if train_idxs.dtype == bool:
        raise TypeError(
            f"'train_idxs' must be a vector of integer slicers, not bools"
        )

    if len(train_idxs.shape) != 1:
        raise ValueError(f"'train_idxs' must be a 1D vector")

    test_idxs = da.array(test_idxs)
    if test_idxs.dtype == bool:
        raise TypeError(
            f"'test_idxs' must be a vector of integer slicers, not bools"
        )

    if len(test_idxs.shape) != 1:
        raise ValueError(f"'test_idxs' must be a 1D vector")


    SPLITS = []
    for _data in data_objects:

        if isinstance(_data, da.core.Array):
            pass
        elif isinstance(_data, ddf.core.DataFrame):
            raise TypeError(f"A dask dataframe is in _fold_splitter, should only "
                f"be dask array")
        elif isinstance(_data, ddf.core.Series):
            raise TypeError(f"A dask series is in _fold_splitter, should "
                f"only be dask array")
        else:
            raise TypeError(f"object of disallowed dtype '{type(_data)}' is in "
                f"fold_splitter()")

        _data_shape = compute(_data.shape, scheduler=scheduler)[0]

        assert len(_data_shape) in [1, 2], f"data objects must be 1-D or 2-D"

        # validate data_objects rows == sum of folds' rows ** * ** * ** * ** *
        # removed this 24_07_13 to allow for splits that do not use the entire
        # dataset
        # assert compute(len(train_idxs) + len(test_idxs), scheduler=scheduler)[0] == \
        #     _data_shape[0], \
        #     "fold_splitter(): (len(train_idxs) + len(test_idxs)) != _data.shape[0]"
        # END validate data_objects rows == sum of folds' rows ** * ** * ** * *

        _data_train, _data_test = _data[train_idxs], _data[test_idxs]

        # IF USING DASK OBJECTS, ENTIRE OBJECT WOULD HAVE TO BE READ OFF DISK TO
        # GET SHAPE, SO ONLY DO THIS FOR EAGER OBJECTS, BUT KEEP IN DASK FOR POSTERITY
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







