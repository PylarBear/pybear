# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

# demo_test incidentally handles testing of all autogridsearch_wrapper
# functionality except fit() (because demo bypasses fit().) This test
# module handles arg/kwarg validation at the highest level.



import pytest
import numpy as np
from pybear.model_selection import autogridsearch_wrapper
from sklearn.model_selection import GridSearchCV as skl_GridSearchCV
from sklearn.linear_model import LogisticRegression as skl_logistic



class TestAGSCV_Generic:

    #         estimator,
    #         params: ParamsType,
    #         total_passes: int = 5,
    #         total_passes_is_hard: bool = False,
    #         max_shifts: Union[None, int] = None,
    #         agscv_verbose: bool = False,
    #         **parent_gscv_kwargs


    @pytest.fixture
    def good_sk_logistic_params(self):
        return {
            'C': [np.logspace(-5,5,6), [6,6,6], 'soft_float'],
            'l1_ratio': [np.linspace(0,1,6), [6,6,6], 'hard_float'],
            'solver': [['saga', 'lbfgs'], 2, 'string']
        }

    @pytest.fixture
    def AutoGridSearch(self):
        return autogridsearch_wrapper(skl_GridSearchCV)


    # params ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    @pytest.mark.parametrize('junk_params',
        (2, np.pi, False, None, [1,2], (1,2), {1,2}, min, lambda x: x, 'junk')
    )
    def test_rejects_junk_params(self, AutoGridSearch, junk_params):
        with pytest.raises(TypeError):
            AutoGridSearch(
                skl_logistic(),
                junk_params
            )

    @pytest.mark.parametrize('bad_params',
        ({'a': ['more_junk']}, {0: [1,2,3,4]}, {'junk': [1, 2, 'what?!']},
         {'b': {1,2,3,4}}, {'qqq': {'rrr': [[1,2,3], 3, 'string']}})
    )
    def test_rejects_bad_params(self, AutoGridSearch, bad_params):
        with pytest.raises(Exception):
            AutoGridSearch(
                skl_logistic(),
                bad_params
            )

    def test_accepts_good_params(self, AutoGridSearch, good_sk_logistic_params):
        AutoGridSearch(
            skl_logistic(),
            good_sk_logistic_params
        )

    # END params ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *



    # total_passes ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @pytest.mark.parametrize('junk_passes',
        (True, None, np.pi, 'junk', min, [1,], (1,), {1,2}, lambda x: x)
    )
    def test_rejects_junk_total_passes(self, junk_passes, AutoGridSearch,
                                       good_sk_logistic_params):
        with pytest.raises(TypeError):
            AutoGridSearch(
                skl_logistic(),
                good_sk_logistic_params,
                total_passes=junk_passes
            )

    @pytest.mark.parametrize('bad_tp', (-1, 0))
    def test_rejects_bad_total_passes(self, bad_tp, AutoGridSearch,
                                      good_sk_logistic_params):
        with pytest.raises(ValueError):
            AutoGridSearch(
                skl_logistic(),
                good_sk_logistic_params,
                total_passes=bad_tp
            )

    @pytest.mark.parametrize('good_tp', (3,))  # must match good_sk_params
    def test_accepts_good_total_passes(self, good_tp, AutoGridSearch,
                                      good_sk_logistic_params):
        assert AutoGridSearch(
            skl_logistic(),
            good_sk_logistic_params,
            total_passes=good_tp
        ).total_passes == good_tp

    # END total_passes ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *



    # tpih ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # takes bool() of input... anything not None, False, 0 is True
    @pytest.mark.parametrize('_tpih',
        (1, 2, np.pi, True, 'junk', min, [1,], (1,), {1,2}, lambda x: x)
    )
    def test_tpih_returns_true(self, _tpih, AutoGridSearch,
                                       good_sk_logistic_params):
        assert AutoGridSearch(
            skl_logistic(),
            good_sk_logistic_params,
            total_passes_is_hard=_tpih
        ).total_passes_is_hard is True


    @pytest.mark.parametrize('_tpih', (0, None, False))
    def test_tpih_returns_false(self, _tpih, AutoGridSearch,
                                      good_sk_logistic_params):
        assert AutoGridSearch(
            skl_logistic(),
            good_sk_logistic_params,
            agscv_verbose=_tpih
        ).total_passes_is_hard is False

    # END tpih ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **



    # max_shifts ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('junk_max_shifts',
        (np.pi, 'junk', min, [1,], (1,), {1,2}, lambda x: x)
    )
    def test_rejects_junk_max_shifts(self, junk_max_shifts, AutoGridSearch,
                                       good_sk_logistic_params):
        with pytest.raises(TypeError):
            AutoGridSearch(
                skl_logistic(),
                good_sk_logistic_params,
                max_shifts=junk_max_shifts
            )

    @pytest.mark.parametrize('bad_max_shifts', (-1, 0))
    def test_rejects_bad_max_shifts(self, bad_max_shifts, AutoGridSearch,
                                      good_sk_logistic_params):
        with pytest.raises(ValueError):
            AutoGridSearch(
                skl_logistic(),
                good_sk_logistic_params,
                max_shifts=bad_max_shifts
            )

    @pytest.mark.parametrize('good_max_shifts', (None, 1, 3))
    def test_accepts_good_max_shifts(self, good_max_shifts, AutoGridSearch,
                                      good_sk_logistic_params):
        assert AutoGridSearch(
            skl_logistic(),
            good_sk_logistic_params,
            max_shifts=good_max_shifts
        ).max_shifts == (good_max_shifts or 100)

    # END max_shifts ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # agscv_verbose ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # takes bool() of input... anything not None, False, 0 is True

    @pytest.mark.parametrize('_verbose',
        (1, 2, np.pi, True, 'junk', min, [1,], (1,), {1,2}, lambda x: x)
    )
    def test_verbose_returns_true(self, _verbose, AutoGridSearch,
                                       good_sk_logistic_params):
        assert AutoGridSearch(
            skl_logistic(),
            good_sk_logistic_params,
            agscv_verbose=_verbose
        ).agscv_verbose is True


    @pytest.mark.parametrize('_verbose', (0, None, False))
    def test_verbose_returns_false(self, _verbose, AutoGridSearch,
                                      good_sk_logistic_params):
        assert AutoGridSearch(
            skl_logistic(),
            good_sk_logistic_params,
            agscv_verbose=_verbose
        ).agscv_verbose is False


    # END agscv_verbose ** * ** * ** * ** * ** * ** * ** * ** * ** * **














