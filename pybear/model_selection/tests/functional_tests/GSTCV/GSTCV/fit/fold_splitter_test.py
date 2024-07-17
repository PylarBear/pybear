# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np
import pandas as pd
import dask.array as da

from model_selection.GSTCV._GSTCV._fit._fold_splitter import _fold_splitter


class TestSKFoldSplitter:

    # def _fold_splitter(
    #         train_idxs: KFoldType,
    #         test_idxs: KFoldType,
    #         *data_objects: Union[XWIPType, YWIPType]
    #     ):


    @pytest.mark.parametrize('bad_data_object', (1, 3.14, True, False, None,
        'junk', min, [0,1], (0,1), {0,1}, {'a':1}, lambda x: x,
        da.random.randint(0,10,(5,3)), pd.DataFrame())
    )
    def test_rejects_everything_not_np_array(self, bad_data_object):

        with pytest.raises(TypeError):
            _fold_splitter(
                [0,2,4],
                [1,3],
                bad_data_object
            )


    @pytest.mark.parametrize('bad_data_object',
        (
            np.random.randint(0, 10, (3, 3, 3)),
            np.random.randint(0, 10, (3, 3, 3, 3)),
        )
    )
    def test_rejects_bad_shape(self, bad_data_object):

        with pytest.raises(AssertionError):
            _fold_splitter(
                [0,2,4],
                [1,3],
                bad_data_object
            )


    def test_accuracy(self):

        out = _fold_splitter(
            [0,2,4],
            [1,3],
            np.array([1,2,3,4,5])
        )

        assert np.array_equiv(out[0], [1,3,5])
        assert np.array_equiv(out[1], [2,4])

        mask_train = np.random.choice(
            range(1_000_000), (750_000,), replace=False
        )
        _ = np.ones(1_000_000).astype(bool)
        _[mask_train] = False
        mask_test = np.arange(1_000_000)[_]

        in1 = np.random.randint(0, 10, (1_000_000, 2))
        in2 = np.random.randint(0, 2, (1_000_000, ))

        out = _fold_splitter(
            mask_train,
            mask_test,
            in1,
            in2
        )

        assert np.array_equiv(out[0], in1[mask_train, :])
        assert np.array_equiv(out[1], in1[mask_test, :])
        assert np.array_equiv(out[2], in2[mask_train])
        assert np.array_equiv(out[3], in2[mask_test])
















