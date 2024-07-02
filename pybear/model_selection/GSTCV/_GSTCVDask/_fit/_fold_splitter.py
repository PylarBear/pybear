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
    DaskKFoldType, XDaskWIPType, YDaskWIPType, SchedulerType
)

def _fold_splitter(
        train_idxs: DaskKFoldType,
        test_idxs: DaskKFoldType,
        *data_objects: Union[XDaskWIPType, YDaskWIPType],
        scheduler: SchedulerType=None
    ):


    train_idxs = da.array(train_idxs)
    if train_idxs.dtype == bool:
        raise TypeError(f"'train_idxs' must be a vector of integer slicers, not bools")

    if len(train_idxs.shape) != 1:
        raise ValueError(f"'train_idxs' must be a 1D vector")

    test_idxs = da.array(test_idxs)
    if test_idxs.dtype == bool:
        raise TypeError(f"'test_idxs' must be a vector of integer slicers, not bools")

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

        _data_train, _data_test = _data[train_idxs], _data[test_idxs]

        # validate data_objects rows == sum of folds' rows ** * ** * ** * ** *
        assert compute(len(train_idxs) + len(test_idxs), scheduler=scheduler)[0] == \
            _data_shape[0], \
            "fold_splitter(): (len(train_idxs) + len(test_idxs)) != _data.shape[0]"
        # END validate data_objects rows == sum of folds' rows ** * ** * ** * *

        # IF USING DASK OBJECTS, ENTIRE OBJECT WOULD HAVE TO BE READ OFF DISK TO
        # GET SHAPE, SO ONLY DO THIS FOR EAGER OBJECTS, BUT KEEP IN DASK FOR POSTERITY
        # train_shape = _data_train.shape
        # test_shape = _data_test.shape
        #
        # assert train_shape[0] == len(train_idxs), \
        #     "fold_splitter(): _data_train.shape[0] != len(train_idxs)"
        # assert test_shape[0] == len(test_idxs), \
        #     "fold_splitter(): _data_test.shape[0] != len(test_idxs)"
        # if len(train_shape) == 2:
        #     assert train_shape[1] == _data.shape[1], \
        #         "fold_splitter(): _data_train.shape[1] != _data.shape[1]"
        #     assert test_shape[1] == _data.shape[1], \
        #         "fold_splitter(): _data_test.shape[1] != _data.shape[1]"
        #
        # del train_shape, test_shape

        SPLITS += [_data_train, _data_test]


    assert len(SPLITS) == 2 * len(data_objects), \
        "fold_splitter(): len(SPLITS) != 2*len(data_objects)"


    return tuple(SPLITS)







