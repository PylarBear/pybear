# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np
import dask.array as da
from sklearn.model_selection import KFold as sk_KFold
from dask_ml.model_selection import KFold as dask_KFold

from pybear.model_selection.GSTCV._GSTCVDask._fit._estimator_fit_params_helper \
    import _estimator_fit_params_helper



class TestEstimatorFitParamsHelper:

    # def _estimator_fit_params_helper(
    #     _data_len: int,
    #     _fit_params: dict[str, Any],
    #     _KFOLD: DaskKFoldType
    # ) -> dict[int, dict[str, Any]]:


    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @staticmethod
    @pytest.fixture
    def good_dask_fit_params(_rows):
        return {
            'sample_weight': da.random.uniform(0, 1, _rows),
            'fake_sample_weight': da.random.uniform(0, 1, _rows // 2),
            'made_up_param_1':  'something_else',
            'made_up_param_2': False,
            'some_other_param_1': {'abc': 123}
        }


    @staticmethod
    @pytest.fixture
    def good_sk_fit_params(good_dask_fit_params):
        # use dask fit params to make sk, this is needed because the
        # vectors in fit params must be equal
        __ = {}
        for param, value in good_dask_fit_params.items():
            try:
                __[param] = value.compute()
            except:
                __[param] = value

        return __


    @staticmethod
    @pytest.fixture
    def good_sk_kfold(standard_cv_int, X_da, y_da):
        return list(sk_KFold(n_splits=standard_cv_int).split(X_da, y_da))


    @staticmethod
    @pytest.fixture
    def good_dask_kfold(standard_cv_int, X_da, y_da):
        return list(dask_KFold(n_splits=standard_cv_int).split(X_da, y_da))


    @staticmethod
    @pytest.fixture
    def exp_dask_helper_output(_rows, good_dask_fit_params, good_dask_kfold):

        dask_helper = {}

        for idx, (train_idxs, test_idxs) in enumerate(good_dask_kfold):

            dask_helper[idx] = {}

            for k, v in good_dask_fit_params.items():

                try:
                    iter(v)
                    if isinstance(v, (dict, str)):
                        raise

                    if len(v) != _rows:
                        raise

                except:
                    dask_helper[idx][k] = v
                    continue

                dask_helper[idx][k] = v.copy()[train_idxs]

        return dask_helper

    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *



    # test validation of args ** * ** * ** * ** * ** * ** * ** * ** * **
    @pytest.mark.parametrize('bad_data_len',
        (-3.14, -1, 0, True, None, 'junk', [0,1], (1,2), {'a': 1}, min,
         lambda x: x)
    )
    def test_data_len_rejects_not_pos_int(
        self, bad_data_len, good_sk_fit_params, good_dask_fit_params,
        good_sk_kfold, good_dask_kfold
    ):

        with pytest.raises(TypeError):
            _estimator_fit_params_helper(
                bad_data_len, good_sk_fit_params, good_sk_kfold
            )

        with pytest.raises(TypeError):
            _estimator_fit_params_helper(
                bad_data_len, good_dask_fit_params, good_dask_kfold
            )


    @pytest.mark.parametrize('bad_fit_params',
        (-3.14, -1, 0, True, None, 'junk', [0,1], (1,2), min, lambda x: x)
    )
    def test_fit_params_rejects_not_dict(
        self, _rows, bad_fit_params, good_sk_kfold, good_dask_kfold
    ):

        with pytest.raises(AssertionError):
            _estimator_fit_params_helper(
                _rows, bad_fit_params, good_sk_kfold
            )

        with pytest.raises(AssertionError):
            _estimator_fit_params_helper(
                _rows, bad_fit_params, good_dask_kfold
            )



    @pytest.mark.parametrize('bad_kfold',
        (-3.14, -1, 0, True, None, 'junk', [0,1], (1,2), {'a': 1}, min,
         lambda x: x)
    )
    def test_kfold_rejects_not_list_of_tuples(
        self, _rows, good_sk_fit_params, good_dask_fit_params, bad_kfold
    ):

        with pytest.raises(AssertionError):
            _estimator_fit_params_helper(
                _rows, good_sk_fit_params, bad_kfold
            )

        with pytest.raises(AssertionError):
            _estimator_fit_params_helper(
                _rows, good_dask_fit_params, bad_kfold
            )

    # END test validation of args ** * ** * ** * ** * ** * ** * ** * ** *




    # test accuracy ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    @pytest.mark.parametrize('kfold_type', ('dask', 'sklearn'))
    @pytest.mark.parametrize('fit_params_type', ('dask', 'sklearn'))
    def test_accuracy(
        self, _rows, good_sk_fit_params, good_sk_kfold,
        good_dask_fit_params, good_dask_kfold, exp_dask_helper_output,
        kfold_type, fit_params_type
    ):

        if fit_params_type=='dask':
            _fit_params = good_dask_fit_params
        elif fit_params_type=='sklearn':
            _fit_params = good_sk_fit_params

        if kfold_type=='dask':
            _kfold = good_dask_kfold
        elif kfold_type=='sklearn':
            _kfold = good_sk_kfold

        out = _estimator_fit_params_helper(_rows, _fit_params, _kfold)

        for f_idx, exp_fold_fit_param_dict in exp_dask_helper_output.items():

            for param, exp_value in exp_fold_fit_param_dict.items():

                # make the output of helper if array always be dask, no
                # matter what was given. later on, that vector would be
                # applied, in some way, (consider logistic 'sample_weight')
                # to X, which always must be dask. dont want to risk
                # trying to operate on a dask with a non-dask.
                if isinstance(exp_value, da.core.Array):  # edho - if array, always dask
                    # if exp_value is an array, out must always be a dask array
                    assert isinstance(out[f_idx][param], da.core.Array)

                # convert everything to numpy for easy len & array_equiv
                try:
                    _ = out[f_idx][param].compute()
                except:
                    _ = out[f_idx][param]

                try:
                    __ = exp_value.compute()
                except:
                    __ = exp_value

                if isinstance(_, np.ndarray): # if was any array, must be np now
                    assert len(_) < _rows
                    assert len(__) < _rows
                    assert np.array_equiv(_, __)
                else:
                    assert _ == __


    def test_accuracy_empty(self, _rows, good_dask_kfold):

        out = _estimator_fit_params_helper(
            _rows,
            {},
            good_dask_kfold
        )

        assert np.array_equiv(list(out), list(range(len(good_dask_kfold))))

        for idx, fit_params in out.items():
            assert fit_params == {}


