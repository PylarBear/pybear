# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest
import inspect

import numpy as np
import dask.array as da

from pybear.model_selection.GSTCV._GSTCVDask._fit._get_kfold import _get_kfold



class TestGetDaskKFold:

    # def _get_kfold(
    #     _X: XDaskWIPType,
    #     _n_splits: int,
    #     _iid: bool,
    #     _verbose: int,
    #     _y: Optional[YDaskWIPType] = None
    # ) -> Generator[DaskKFoldType, None, None]:

    # X, y must both be da.array
    # AS OF 25_04_28 ONLY DASK ARRAYS CAN BE PASSED TO
    # dask_KFOLD (NOT np, pd.DF, dask.DF)
    # see dask_kfold_input_test in functional_tests folder

    # *** IMPORTANT!!!
    # This function can be called multiple times within a single param grid
    # permutation, first to fit, again to get test score, then again if
    # return_train_score. Therefore, it must return the same indices for
    # each call. The only things that should cause indices to be different
    # are n_splits and the number of rows in _X. Since this is dask KFold,
    # there is the wildcard of the 'iid' setting. If iid is False -- meaning
    # the data is known to have some non-random grouping along axis 0 --
    # via the 'shuffle' argument KFold will generate indices that sample
    # across chunks to randomize the data in the splits. In that case, fix
    # the random_state parameter to make selection repeatable. If iid is
    # True, shuffle is False, random_state can be None, and the splits
    # should be repeatable.

    # fixtures -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    @staticmethod
    @pytest.fixture(scope='function')
    def _base_X():
        return da.random.randint(0, 10, (100, 30))


    @staticmethod
    @pytest.fixture(scope='function')
    def _base_y():
        return da.random.randint(0, 10, (100, 1))


    # pizza 25_04_29 the reason this doesnt have a _format_helper
    # is because dask_ml KFold can only take da.array. but if _format_helper
    # ends up in conftest then it may be helpful to use to standardize this
    # with sk get_kfold

    # END fixtures -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    @pytest.mark.parametrize('_junk_X',
        (-2.7, -1, 0, 1, 2.7, None, 'str', lambda x:x)
    )
    @pytest.mark.parametrize('_junk_y',
        (-2.7, -1, 0, 1, 2.7, None, 'str', lambda x:x)
    )
    def test_X_y_rejects_junk(self, _junk_X, _junk_y):

        # this is raised by dask_ml.KFold, let it raise whatever
        with pytest.raises(Exception):
            list(_get_kfold(
                _junk_X,
                _n_splits=3,
                _iid=True,
                _verbose=0,
                _y=_junk_y
            ))


    @pytest.mark.parametrize(f'junk_n_splits',
        (-1, 0, 1, 3.14, None, min, 'junk', [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_n_splits(self, _base_X, _base_y, junk_n_splits):
        with pytest.raises(AssertionError):
            _get_kfold(
                _base_X,
                _n_splits=junk_n_splits,
                _iid=True,
                _verbose=0,
                _y=_base_y
            )


    @pytest.mark.parametrize(f'junk_iid',
        (0, 1, 3.14, None, min, 'junk', [0, 1], (0, 1), {0, 1}, {'a': 1},
         lambda x: x)
    )
    def test_rejects_non_bool_iid(self, _base_X, _base_y, junk_iid):
        with pytest.raises(AssertionError):
            _get_kfold(
                _base_X,
                _n_splits=3,
                _iid=junk_iid,
                _verbose=0,
                _y=_base_y
            )


    @pytest.mark.parametrize(f'junk_verbose',
        (-1, None, min, 'junk', [0, 1], (0, 1), {0, 1}, {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_verbose(self, _base_X, _base_y, junk_verbose):
        with pytest.raises(AssertionError):
            _get_kfold(
                _base_X,
                _n_splits=3,
                _iid=True,
                _verbose=junk_verbose,
                _y=_base_y
            )

    # END validation -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    def test_da_returns_gen_of_das(self, _base_X, _base_y):

        # and second call returns same thing as the first
        out1 = _get_kfold(
            _base_X,
            _n_splits=3,
            _iid=True,
            _verbose=0,
            _y=_base_y
        )

        assert inspect.isgenerator(out1)

        out1_list = list(out1)

        for (train_idxs, test_idxs) in out1:

            assert isinstance(train_idxs, da.core.Array)
            assert isinstance(test_idxs, da.core.Array)

            assert train_idxs.min() >= 0
            assert train_idxs.max() < _base_X.shape[0]

            assert test_idxs.min() >= 0
            assert test_idxs.max() < _base_X.shape[0]


        # and second call returns same as the first
        out2 = _get_kfold(
            _base_X,
            _n_splits=3,
            _iid=True,
            _verbose=0,
            _y=_base_y
        )


        for idx, (train_idxs2, test_idxs2) in enumerate(out2):

            assert isinstance(train_idxs2, da.core.Array)
            assert isinstance(test_idxs2, da.core.Array)

            assert train_idxs2.min() >= 0
            assert train_idxs2.max() < _base_X.shape[0]

            assert test_idxs2.min() >= 0
            assert test_idxs2.max() < _base_X.shape[0]

            assert np.array_equiv(out1_list[idx][0], train_idxs2)
            assert np.array_equiv(out1_list[idx][1], test_idxs2)




