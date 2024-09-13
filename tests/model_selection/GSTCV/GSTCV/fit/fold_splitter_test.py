# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np
import pandas as pd
import dask.array as da


from pybear.model_selection.GSTCV._GSTCV._fit._fold_splitter import _fold_splitter



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


    def test_accuracy(self, X_np, y_np, _rows):

        out = _fold_splitter(
            [0,2,4],
            [1,3],
            np.array([1,2,3,4,5])
        )

        assert isinstance(out[0], np.ndarray)
        assert np.array_equiv(out[0], [1,3,5])
        assert isinstance(out[1], np.ndarray)
        assert np.array_equiv(out[1], [2,4])

        mask_train = np.random.choice(
            range(_rows), (int(0.75 * _rows), ), replace=False
        )
        _ = np.ones(_rows).astype(bool)
        _[mask_train] = False
        mask_test = np.arange(_rows)[_]

        out = _fold_splitter(
            mask_train,
            mask_test,
            X_np,
            y_np
        )

        assert isinstance(out[0], np.ndarray)
        assert np.array_equiv(out[0], X_np[mask_train, :])
        assert isinstance(out[1], np.ndarray)
        assert np.array_equiv(out[1], X_np[mask_test, :])
        assert isinstance(out[2], np.ndarray)
        assert np.array_equiv(out[2], y_np[mask_train])
        assert isinstance(out[3], np.ndarray)
        assert np.array_equiv(out[3], y_np[mask_test])
















