# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



# This test module handles arg/kwarg _validation at the highest level.



import pytest
import numpy as np
from pybear.model_selection import autogridsearch_wrapper
from sklearn.model_selection import GridSearchCV as skl_GridSearchCV
from sklearn.linear_model import LogisticRegression as sk_Logistic



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
            'solver': [['saga', 'lbfgs'], 2, 'string'],
            'fit_intercept': [[True, False], 2, 'bool']
        }

    @pytest.fixture
    def AutoGridSearch(self):
        return autogridsearch_wrapper(skl_GridSearchCV)

    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # parent GSCV ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @pytest.mark.parametrize('non_class',
        (0, 1, 3.14, [1,2], (1,2), {1,2}, {'a':1}, 'junk', lambda x: x)
    )
    def test_rejects_anything_not_an_estimator(
        self, AutoGridSearch, good_sk_logistic_params, non_class
    ):
        # this is raised by the parent GSCV, let it raise whatever
        # the parent GSCV wont check inputs until u try to fit
        with pytest.raises(Exception):
            AutoGridSearch(
                estimator=non_class,
                params=good_sk_logistic_params
            ).fit(np.random.uniform(0,1,(20,10)), np.random.randint(0,2,(20,)))


    def test_invalid_estimator(
        self, AutoGridSearch, good_sk_logistic_params
    ):

        class weird_estimator:

            def __init__(cls, crazy_param):
                cls.crazy_param = crazy_param

            def train(cls):
                return cls

            def run(cls):
                return cls.crazy_param


        # this is raised by the parent GSCV, let it raise whatever
        # the parent GSCV wont check inputs until u try to fit
        with pytest.raises(Exception):
            AutoGridSearch(
                estimator=weird_estimator(crazy_param=float('inf')),
                params={'crazy_param': [[True, False], 2, 'bool']}
            ).fit(np.random.uniform(0,1,(20,10)), np.random.randint(0,2,(20,)))

        del weird_estimator


    def test_rejects_bad_sklearn_GSCV_kwargs(
        self, AutoGridSearch, good_sk_logistic_params
    ):

        # this is raised by the parent GSCV, let it raise whatever
        # the parent GSCV wont check inputs until u try to fit
        with pytest.raises(Exception):
            AutoGridSearch(
                estimator=sk_Logistic(),
                params=good_sk_logistic_params,
                aaa=True,
                bbb=1.5
            ).fit(np.random.uniform(0,1,(20,10)), np.random.randint(0,2,(20,)))


    def test_accepts_good_estimator_and_sklearn_GSCV_kwargs(
        self, AutoGridSearch, good_sk_logistic_params
    ):

        # the parent GSCV wont check inputs until u try to fit
        AutoGridSearch(
            estimator=sk_Logistic(),
            params=good_sk_logistic_params,
            scoring='accuracy',
            n_jobs=-1,
            cv=5
        ).fit(np.random.uniform(0,1,(20,10)), np.random.randint(0,2,(20,)))

    # END parent GSCV ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # params ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @pytest.mark.parametrize('junk_params',
        (2, np.pi, False, None, [1,2], (1,2), {1,2}, min, lambda x: x, 'junk')
    )
    def test_rejects_junk_params(self, AutoGridSearch, junk_params):
        with pytest.raises(TypeError):
            AutoGridSearch(
                sk_Logistic(),
                junk_params
            )

    @pytest.mark.parametrize('bad_params',
        ({'a': ['more_junk']}, {0: [1,2,3,4]}, {'junk': [1, 2, 'what?!']},
         {'b': {1,2,3,4}}, {'qqq': {'rrr': [[1,2,3], 3, 'string']}})
    )
    def test_rejects_bad_params(self, AutoGridSearch, bad_params):
        with pytest.raises(Exception):
            AutoGridSearch(
                sk_Logistic(),
                bad_params
            )

    def test_accepts_good_params(self, AutoGridSearch, good_sk_logistic_params):
        AutoGridSearch(
            sk_Logistic(),
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
                sk_Logistic(),
                good_sk_logistic_params,
                total_passes=junk_passes
            )

    @pytest.mark.parametrize('bad_tp', (-1, 0))
    def test_rejects_bad_total_passes(self, bad_tp, AutoGridSearch,
                                      good_sk_logistic_params):
        with pytest.raises(ValueError):
            AutoGridSearch(
                sk_Logistic(),
                good_sk_logistic_params,
                total_passes=bad_tp
            )

    @pytest.mark.parametrize('good_tp', (3,))  # must match good_sk_params
    def test_accepts_good_total_passes(self, good_tp, AutoGridSearch,
                                      good_sk_logistic_params):
        assert AutoGridSearch(
            sk_Logistic(),
            good_sk_logistic_params,
            total_passes=good_tp
        ).total_passes == good_tp

    # END total_passes ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *



    # tpih ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # must be bool
    @pytest.mark.parametrize('_tpih',
        (1, 2, np.pi, None, 'junk', min, [1,], (1,), {1,2}, lambda x: x)
    )
    def test_tpih_rejects_non_bool(
        self, _tpih, AutoGridSearch, good_sk_logistic_params
    ):

        with pytest.raises(TypeError):
            AutoGridSearch(
                sk_Logistic(),
                good_sk_logistic_params,
                total_passes_is_hard=_tpih
            )


    @pytest.mark.parametrize('_tpih', (True, False))
    def test_tpih_accepts_bool(
        self, _tpih, AutoGridSearch, good_sk_logistic_params
    ):
        assert AutoGridSearch(
            sk_Logistic(),
            good_sk_logistic_params,
            total_passes_is_hard=_tpih
        ).total_passes_is_hard is _tpih

    # END tpih ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **



    # max_shifts ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('junk_max_shifts',
        (np.pi, 'junk', min, [1,], (1,), {1,2}, lambda x: x)
    )
    def test_rejects_junk_max_shifts(self, junk_max_shifts, AutoGridSearch,
                                       good_sk_logistic_params):
        with pytest.raises(TypeError):
            AutoGridSearch(
                sk_Logistic(),
                good_sk_logistic_params,
                max_shifts=junk_max_shifts
            )

    @pytest.mark.parametrize('bad_max_shifts', (-1, 0))
    def test_rejects_bad_max_shifts(self, bad_max_shifts, AutoGridSearch,
                                      good_sk_logistic_params):
        with pytest.raises(ValueError):
            AutoGridSearch(
                sk_Logistic(),
                good_sk_logistic_params,
                max_shifts=bad_max_shifts
            )

    @pytest.mark.parametrize('good_max_shifts', (None, 1, 3))
    def test_accepts_good_max_shifts(self, good_max_shifts, AutoGridSearch,
                                      good_sk_logistic_params):
        assert AutoGridSearch(
            sk_Logistic(),
            good_sk_logistic_params,
            max_shifts=good_max_shifts
        ).max_shifts == (good_max_shifts or 100)

    # END max_shifts ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # agscv_verbose ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # must be bool

    @pytest.mark.parametrize('_verbose',
        (1, 2, np.pi, None, 'junk', min, [1,], (1,), {1,2}, lambda x: x)
    )
    def test_verbose_rejects_non_bool(
        self, _verbose, AutoGridSearch, good_sk_logistic_params
    ):

        with pytest.raises(TypeError):
            assert AutoGridSearch(
                sk_Logistic(),
                good_sk_logistic_params,
                agscv_verbose=_verbose
            )


    @pytest.mark.parametrize('_verbose', (True, False))
    def test_verbose_accepts_bool(
        self, _verbose, AutoGridSearch, good_sk_logistic_params
    ):
        assert AutoGridSearch(
            sk_Logistic(),
            good_sk_logistic_params,
            agscv_verbose=_verbose
        ).agscv_verbose is _verbose


    # END agscv_verbose ** * ** * ** * ** * ** * ** * ** * ** * ** * **














