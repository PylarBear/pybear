# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np

from sklearn.model_selection import KFold as sk_KFold

from pybear.model_selection.GSTCV._GSTCV._fit._estimator_fit_params_helper \
    import _estimator_fit_params_helper



class TestEstimatorFitParamsHelper:

    #      def _estimator_fit_params_helper(
    #         data_len: int,
    #         fit_params: dict[str, any],
    #         KFOLD: list[tuple[T, T]]
    # ) -> dict[int, dict[str, any]]:


    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    # pizza see if u can convert everything in here to X_np & y_np
    @staticmethod
    @pytest.fixture
    def good_data_len():
        return 10


    @staticmethod
    @pytest.fixture
    def good_sk_fit_params(good_data_len):
        return {
            'sample_weight': np.random.uniform(0, 1, good_data_len),
            'fake_sample_weight': np.random.uniform(0, 1, good_data_len//2),
            'made_up_param_1':  'something',
            'made_up_param_2': True,
            'some_other_param_1': {'abc': 123}
        }


    @staticmethod
    @pytest.fixture
    def good_sk_kfold(standard_cv_int, good_data_len):
        return list(
            sk_KFold(n_splits=standard_cv_int).split(
                np.random.randint(0,10,(good_data_len, 5)),
                np.random.randint(0,2,(good_data_len,))
            )
        )


    @staticmethod
    @pytest.fixture
    def exp_sk_helper_output(good_data_len, good_sk_fit_params, good_sk_kfold):

        sk_helper = {}

        for idx, (train_idxs, test_idxs) in enumerate(good_sk_kfold):

            sk_helper[idx] = {}

            for k, v in good_sk_fit_params.items():

                try:
                    iter(v)
                    if isinstance(v, (dict, str)):
                        raise

                    if len(v) != good_data_len:
                        raise

                    np.array(list(v))

                except:
                    sk_helper[idx][k] = v
                    continue

                sk_helper[idx][k] = v.copy()[train_idxs]

        return sk_helper

    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *



    # test validation of args ** * ** * ** * ** * ** * ** * ** * ** * **
    @pytest.mark.parametrize('bad_data_len',
        (-3.14, -1, 0, True, None, 'junk', [0,1], (1,2), {'a': 1}, min,
         lambda x: x)
    )
    def test_data_len_rejects_not_pos_int(
        self, bad_data_len, good_sk_fit_params, good_sk_kfold
    ):

        with pytest.raises(TypeError):
            _estimator_fit_params_helper(
                bad_data_len, good_sk_fit_params, good_sk_kfold
            )


    @pytest.mark.parametrize('bad_fit_params',
        (-3.14, -1, 0, True, None, 'junk', [0,1], (1,2), min, lambda x: x)
    )
    def test_fit_params_rejects_not_dict(
        self, good_data_len, bad_fit_params, good_sk_kfold
    ):

        with pytest.raises(AssertionError):
            _estimator_fit_params_helper(
                good_data_len, bad_fit_params, good_sk_kfold
            )


    @pytest.mark.parametrize('bad_kfold',
        (-3.14, -1, 0, True, None, 'junk', [0,1], (1,2), {'a': 1}, min,
         lambda x: x)
    )
    def test_kfold_rejects_not_list_of_tuples(
        self, good_data_len, good_sk_fit_params, bad_kfold
    ):

        with pytest.raises(AssertionError):
            _estimator_fit_params_helper(
                good_data_len, good_sk_fit_params, bad_kfold
            )


    # END test validation of args ** * ** * ** * ** * ** * ** * ** * ** *




    # test accuracy ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    def test_sk_accuracy(
        self, good_data_len, good_sk_fit_params, good_sk_kfold, exp_sk_helper_output
    ):

        out = _estimator_fit_params_helper(
            good_data_len, good_sk_fit_params, good_sk_kfold
        )

        for f_idx, exp_fold_fit_param_dict in exp_sk_helper_output.items():

            for param, exp_value in exp_fold_fit_param_dict.items():
                _ = out[f_idx][param]
                __ = exp_value
                if isinstance(exp_value, np.ndarray):
                    assert len(_) < good_data_len
                    assert len(__) < good_data_len
                    assert np.array_equiv(_, __)
                else:
                    assert _ == exp_value


    def test_accuracy_empty(self, good_data_len, good_sk_kfold):

        out = _estimator_fit_params_helper(
            good_data_len,
            {},
            good_sk_kfold
        )

        assert np.array_equiv(list(out), list(range(len(good_sk_kfold))))

        for idx, fit_params in out.items():
            assert fit_params == {}


