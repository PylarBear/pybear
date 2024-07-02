# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest
import inspect

import numpy as np
import dask.array as da
import dask.dataframe as ddf

from model_selection.GSTCV._GSTCVDask._fit._get_kfold import _get_kfold




class TestGetDaskKFold:

    # def _get_kfold(
    #         _X: XDaskWIPType,
    #         _n_splits: int,
    #         _iid: bool,
    #         _verbose: int,
    #         _y: YDaskWIPType = None
    # ) -> Generator[da.core.Array, None, None]:

    # X, y must both be da.array
    # AS OF 24_06_27_09_08_00 ONLY DASK ARRAYS CAN BE PASSED TO
    # dask_KFOLD (NOT np, pd.DF, dask.DF)
    # see dask_kfold_input_test in functional_tests folder

    # important!!! this function can be called multiple times within a
    # single param grid permutation, first to fit and get test score,
    # then again if return_train_score. Therefore, it must return the
    # same indices for each call. The only things that should cause
    # indices to be different are n_splits and the number of rows in X.
    # Shuffle is on if iid is False, therefore random_state state must
    # be set to a constant.


    _X_da = da.random.randint(0, 10, (100, 30))

    _X_junk = ddf.from_array(da.random.randint(0, 10, (100, 30)))

    _y_da = da.random.randint(0, 10, (100, 1))

    _y_junk = ddf.from_array(da.random.randint(0, 10, (100, 1)))


    @pytest.mark.parametrize('_x_name, _X',
        (('_X_da', _X_da), ('_X_junk', _X_junk))
    )
    @pytest.mark.parametrize('_y_name, _y',
        (('_y_da', _y_da), ('_y_junk', _y_junk), ('_y_None', None))
    )
    def test_X_y_must_both_be_da(self, _x_name, _X, _y_name, _y):

        if (_x_name == '_X_junk' or _y_name in '_y_junk'):
            with pytest.raises(TypeError):
                _get_kfold(
                    _X,
                    _n_splits=3,
                    _iid=True,
                    _verbose=0,
                    _y=_y
                )
        else:
            _get_kfold(
                _X,
                _n_splits=3,
                _iid=True,
                _verbose=0,
                _y=_y
            )


    @pytest.mark.parametrize(f'junk_n_splits',
        (-1, 0, 1, 3.14, None, min, 'junk', [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    @pytest.mark.parametrize('_X, _y', ((_X_da, _y_da),))
    def test_rejects_junk_n_splits(self, _X, _y, junk_n_splits):
        with pytest.raises(AssertionError):
            _get_kfold(
                _X,
                _n_splits=junk_n_splits,
                _iid=True,
                _verbose=0,
                _y=_y
            )


    @pytest.mark.parametrize(f'junk_iid',
        (0, 1, 3.14, None, min, 'junk', [0, 1], (0, 1), {0, 1}, {'a': 1}, lambda x: x)
    )
    @pytest.mark.parametrize('_X, _y', ((_X_da, _y_da),))
    def test_rejects_non_bool_iid(self, _X, _y, junk_iid):
        with pytest.raises(AssertionError):
            _get_kfold(
                _X,
                _n_splits=3,
                _iid=junk_iid,
                _verbose=0,
                _y=_y
            )


    @pytest.mark.parametrize(f'junk_verbose',
        (-1, None, min, 'junk', [0, 1], (0, 1), {0, 1}, {'a': 1}, lambda x: x)
    )
    @pytest.mark.parametrize('_X, _y', ((_X_da, _y_da),))
    def test_rejects_junk_verbose(self, _X, _y, junk_verbose):
        with pytest.raises(AssertionError):
            _get_kfold(
                _X,
                _n_splits=3,
                _iid=True,
                _verbose=junk_verbose,
                _y=_y
            )


    @pytest.mark.parametrize('_X_da', (_X_da,))
    @pytest.mark.parametrize('_y_da', (_y_da,))
    def test_da_returns_gen_of_das(self, _X_da, _y_da):

        # and second call returns same thing as the first
        out1 = _get_kfold(
            _X_da,
            _n_splits=3,
            _iid=True,
            _verbose=0,
            _y=_y_da
        )

        assert inspect.isgenerator(out1)

        out1_list = list(out1)

        for (train_idxs, test_idxs) in out1:

            assert isinstance(train_idxs, da.core.Array)
            assert isinstance(test_idxs, da.core.Array)

            assert train_idxs.min() >= 0
            assert train_idxs.max() < _X_da.shape[0]

            assert test_idxs.min() >= 0
            assert test_idxs.max() < _X_da.shape[0]


        out2 = _get_kfold(
            _X_da,
            _n_splits=3,
            _iid=True,
            _verbose=0,
            _y=_y_da
        )


        for idx, (train_idxs2, test_idxs2) in enumerate(out2):

            assert isinstance(train_idxs2, da.core.Array)
            assert isinstance(test_idxs2, da.core.Array)

            assert train_idxs2.min() >= 0
            assert train_idxs2.max() < _X_da.shape[0]

            assert test_idxs2.min() >= 0
            assert test_idxs2.max() < _X_da.shape[0]

            assert np.array_equiv(out1_list[idx][0], train_idxs2)
            assert np.array_equiv(out1_list[idx][1], test_idxs2)









































