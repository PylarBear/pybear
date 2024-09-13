# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as ddf

from pybear.model_selection.GSTCV._GSTCVDask._fit._fold_splitter import \
    _fold_splitter



class TestFoldSplitter:

    # def _fold_splitter(
    #         train_idxs: KFoldType,
    #         test_idxs: KFoldType,
    #         *data_objects: Union[XWIPType, YWIPType],
    #         scheduler: Scheduler=None
    #     ):

    @pytest.mark.parametrize('bad_data_object', (1, 3.14, True, False, None,
        'junk', min, [0,1], (0,1), {0,1}, {'a':1}, lambda x: x,
        np.random.randint(0,10,(5,3)),
        ddf.from_pandas(pd.DataFrame(), npartitions=1))
    )
    def test_rejects_everything_not_dask_array(self, bad_data_object):

        with pytest.raises(TypeError):
            _fold_splitter(
                [0,2,4],
                [1,3],
                bad_data_object
            )


    @pytest.mark.parametrize('bad_data_object',
        (
            da.random.randint(0, 10, (3, 3, 3)),
            da.random.randint(0, 10, (3, 3, 3, 3)),
        )
    )
    def test_rejects_bad_shape(self, bad_data_object):

        with pytest.raises(AssertionError):
            _fold_splitter(
                [0,2,4],
                [1,3],
                bad_data_object
            )


    def test_accuracy(self, X_da, y_da, _rows):

        out = _fold_splitter(
            [0,2,4],
            [1,3],
            da.array([1,2,3,4,5])
        )

        assert isinstance(out[0], da.core.Array)
        assert np.array_equiv(out[0], [1,3,5])
        assert isinstance(out[1], da.core.Array)
        assert np.array_equiv(out[1], [2,4])

        mask_train = da.random.choice(
            range(_rows), (int(0.75 * _rows), ), replace=False
        )
        _ = np.ones(_rows).astype(bool)
        _[mask_train] = False
        mask_test = da.arange(_rows)[_]

        out = _fold_splitter(
            mask_train,
            mask_test,
            X_da,
            y_da
        )

        assert isinstance(out[0], da.core.Array)
        assert np.array_equiv(out[0], X_da[mask_train, :])
        assert isinstance(out[1], da.core.Array)
        assert np.array_equiv(out[1], X_da[mask_test, :])
        assert isinstance(out[2], da.core.Array)
        assert np.array_equiv(out[2], y_da[mask_train])
        assert isinstance(out[3], da.core.Array)
        assert np.array_equiv(out[3], y_da[mask_test])
















