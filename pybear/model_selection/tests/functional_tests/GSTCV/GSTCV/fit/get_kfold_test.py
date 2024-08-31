# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest
import inspect

import numpy as np
import pandas as pd


from model_selection.GSTCV._GSTCV._fit._get_kfold import _get_kfold



class TestSKGetKFold:

    # def _get_kfold(
    #         _X: XSKWIPType,
    #         _y: YSKWIPType,
    #         _n_splits: int,
    #         _verbose: int,
    #         **kfold_params
    #     ) -> Generator[npt.NDArray[int], None, None]:


    # X, y must both be np.array

    # important!!! this function can be called multiple times within a
    # single param grid permutation, first to fit and get test score,
    # then again if return_train_score. Therefore, it must return the
    # same indices for each call. The only things that should cause
    # indices to be different are n_splits and the number of rows in X.
    # Since this is stratified KFold and examples are pulled based on the
    # distribution of y, set random_state state to a constant.


    _X_np = np.random.randint(0, 10, (100, 30))

    _X_junk = pd.DataFrame(_X_np)

    _y_np = np.random.randint(0, 10, (100, 1))

    _y_junk = pd.DataFrame(_y_np)


    @pytest.mark.parametrize('_x_name, _X',
        (('_X_np', _X_np), ('_X_junk', _X_junk), ('_X_junk', None))
    )
    @pytest.mark.parametrize('_y_name, _y',
        (('_y_np', _y_np), ('_y_junk', _y_junk), ('_y_junk', None))
    )
    def test_X_y_must_be_both_np(self, _x_name, _X, _y_name, _y):

        if (_x_name == '_X_junk' or _y_name in '_y_junk'):
            with pytest.raises(TypeError):
                _get_kfold(
                    _X,
                    _y,
                    _n_splits=3,
                    _verbose=0
                )
        else:
            _get_kfold(
                _X,
                _y,
                _n_splits=3,
                _verbose=0
            )


    @pytest.mark.parametrize(f'junk_n_splits',
        (-1, 0, 1, 3.14, True, min, 'junk', [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    @pytest.mark.parametrize('_X, _y', ((_X_np, _y_np),))
    def test_rejects_junk_n_splits(self, _X, _y, junk_n_splits):
        with pytest.raises(AssertionError):
            _get_kfold(
                _X,
                _y,
                _n_splits=junk_n_splits,
                _verbose=0
            )


    @pytest.mark.parametrize(f'junk_verbose',
        (-1, min, 'junk', [0, 1], (0, 1), {0, 1}, {'a': 1}, lambda x: x)
    )
    @pytest.mark.parametrize('_X, _y', ((_X_np, _y_np),))
    def test_rejects_junk_verbose(self, _X, _y, junk_verbose):
        with pytest.raises(AssertionError):
            _get_kfold(
                _X,
                _y,
                _n_splits=3,
                _verbose=junk_verbose
            )


    @pytest.mark.parametrize('_X_np, _y_np', ((_X_np, _y_np),))
    def test_np_returns_gen_of_nps(self, _X_np, _y_np):

        out1 = _get_kfold(
            _X_np,
            _y_np,
            _n_splits=3,
            _verbose=0,
        )

        assert inspect.isgenerator(out1)

        out1_list = list(out1)

        for (train_idxs, test_idxs) in out1:

            assert isinstance(train_idxs, np.ndarray)
            assert isinstance(test_idxs, np.ndarray)

            assert train_idxs.min() >= 0
            assert train_idxs.max() < _X_np.shape[0]

            assert test_idxs.min() >= 0
            assert test_idxs.max() < _X_np.shape[0]


        # and second call returns same as the first
        out2 = _get_kfold(
            _X_np,
            _y_np,
            _n_splits=3,
            _verbose=0
        )


        for idx, (train_idxs2, test_idxs2) in enumerate(out2):

            assert isinstance(train_idxs2, np.ndarray)
            assert isinstance(test_idxs2, np.ndarray)

            assert train_idxs2.min() >= 0
            assert train_idxs2.max() < _X_np.shape[0]

            assert test_idxs2.min() >= 0
            assert test_idxs2.max() < _X_np.shape[0]

            assert np.array_equiv(out1_list[idx][0], train_idxs2)
            assert np.array_equiv(out1_list[idx][1], test_idxs2)









































