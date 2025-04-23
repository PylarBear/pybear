# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



# This test module handles arg/kwarg _validation at the highest level.



import pytest
import numpy as np

from pybear.model_selection.autogridsearch.autogridsearch_wrapper import \
    autogridsearch_wrapper

from sklearn.model_selection import GridSearchCV as skl_GridSearchCV
from sklearn.linear_model import LogisticRegression as sk_Logistic

from sklearn.datasets import make_classification




class TestAGSCVValidation:

    #         estimator,
    #         params: ParamsType,
    #         total_passes: int = 5,
    #         total_passes_is_hard: bool = False,
    #         max_shifts: Union[None, int] = None,
    #         agscv_verbose: bool = False,
    #         **parent_gscv_kwargs


    @staticmethod
    @pytest.fixture
    def good_sk_logistic_params():
        return {
            'C': [np.logspace(-5,5,6), 6, 'soft_float'],
            'l1_ratio': [np.linspace(0,1,6), 6, 'hard_float'],
            'solver': [['saga', 'lbfgs'], 2, 'fixed_string'],
            'fit_intercept': [[True, False], 2, 'fixed_bool']
        }


    @staticmethod
    @pytest.fixture
    def AutoGridSearch():
        return autogridsearch_wrapper(skl_GridSearchCV)


    @staticmethod
    @pytest.fixture
    def _X():
        return np.random.uniform(0, 1, (20, 4))


    @staticmethod
    @pytest.fixture
    def _y():
        return np.random.randint(0, 2, (20,))

    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # parent GSCV ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @pytest.mark.parametrize('non_class',
        (0, 1, 3.14, [1,2], (1,2), {1,2}, {'a':1}, 'junk', lambda x: x)
    )
    def test_rejects_anything_not_an_estimator(
        self, AutoGridSearch, good_sk_logistic_params, _X, _y, non_class
    ):
        # this is raised by the parent GSCV, let it raise whatever
        # the parent GSCV wont check inputs until u try to fit
        with pytest.raises(Exception):
            AutoGridSearch(
                estimator=non_class,
                params=good_sk_logistic_params
            ).fit(_X, _y)


    def test_invalid_estimator(
        self, AutoGridSearch, good_sk_logistic_params, _X, _y
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
                params={'crazy_param': [[True, False], 2, 'fixed_bool']}
            ).fit(_X, _y)

        del weird_estimator


    def test_rejects_bad_sklearn_GSCV_kwargs(
        self, AutoGridSearch, good_sk_logistic_params, _X, _y
    ):

        # this is raised by the parent GSCV, let it raise whatever
        # the parent GSCV wont check inputs until u try to fit
        with pytest.raises(Exception):
            AutoGridSearch(
                estimator=sk_Logistic(),
                params=good_sk_logistic_params,
                aaa=True,
                bbb=1.5
            ).fit(_X, _y)

    # END parent GSCV ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # params ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @pytest.mark.parametrize('junk_params',
        (2, np.pi, False, None, [1,2], (1,2), {1,2}, min, lambda x: x, 'junk')
    )
    def test_rejects_junk_params(self, AutoGridSearch, junk_params, _X, _y):
        with pytest.raises(TypeError):
            AutoGridSearch(
                sk_Logistic(),
                junk_params
            ).fit(_X, _y)


    @pytest.mark.parametrize('bad_params',
        ({'a': ['more_junk']}, {0: [1,2,3,4]}, {'junk': [1, 2, 'what?!']},
         {'b': {1,2,3,4}}, {'qqq': {'rrr': [[1,2,3], 3, 'fixed_string']}})
    )
    def test_rejects_bad_params(self, AutoGridSearch, bad_params, _X, _y):
        with pytest.raises(Exception):
            AutoGridSearch(
                sk_Logistic(),
                bad_params
            ).fit(_X, _y)

    # END params ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # total_passes ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @pytest.mark.parametrize('junk_passes',
        (True, None, np.pi, 'junk', min, [1,], (1,), {1,2}, lambda x: x)
    )
    def test_rejects_junk_total_passes(
        self, junk_passes, AutoGridSearch, good_sk_logistic_params, _X, _y
    ):
        with pytest.raises(TypeError):
            AutoGridSearch(
                sk_Logistic(),
                good_sk_logistic_params,
                total_passes=junk_passes
            ).fit(_X, _y)


    @pytest.mark.parametrize('bad_tp', (-1, 0))
    def test_rejects_bad_total_passes(
        self, bad_tp, AutoGridSearch, good_sk_logistic_params, _X, _y
    ):
        with pytest.raises(ValueError):
            AutoGridSearch(
                sk_Logistic(),
                good_sk_logistic_params,
                total_passes=bad_tp
            ).fit(_X, _y)

    # END total_passes ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # tpih ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # must be bool
    @pytest.mark.parametrize('_tpih',
        (1, 2, np.pi, None, 'junk', min, [1,], (1,), {1,2}, lambda x: x)
    )
    def test_tpih_rejects_non_bool(
        self, _tpih, AutoGridSearch, good_sk_logistic_params, _X, _y
    ):

        with pytest.raises(TypeError):
            AutoGridSearch(
                sk_Logistic(),
                good_sk_logistic_params,
                total_passes_is_hard=_tpih
            ).fit(_X, _y)

    # END tpih ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    # max_shifts ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('junk_max_shifts',
        (np.pi, 'junk', min, [1,], (1,), {1,2}, lambda x: x)
    )
    def test_rejects_junk_max_shifts(
        self, junk_max_shifts, AutoGridSearch, good_sk_logistic_params, _X, _y
    ):
        with pytest.raises(TypeError):
            AutoGridSearch(
                sk_Logistic(),
                good_sk_logistic_params,
                max_shifts=junk_max_shifts
            ).fit(_X, _y)


    @pytest.mark.parametrize('bad_max_shifts', (-1, 0))
    def test_rejects_bad_max_shifts(
        self, bad_max_shifts, AutoGridSearch, good_sk_logistic_params, _X, _y
    ):
        with pytest.raises(ValueError):
            AutoGridSearch(
                sk_Logistic(),
                good_sk_logistic_params,
                max_shifts=bad_max_shifts
            ).fit(_X, _y)

    # END max_shifts ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # agscv_verbose ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # must be bool

    @pytest.mark.parametrize('_verbose',
        (1, 2, np.pi, None, 'junk', min, [1,], (1,), {1,2}, lambda x: x)
    )
    def test_verbose_rejects_non_bool(
        self, _verbose, AutoGridSearch, good_sk_logistic_params, _X, _y
    ):

        with pytest.raises(TypeError):
            AutoGridSearch(
                sk_Logistic(),
                good_sk_logistic_params,
                agscv_verbose=_verbose
            ).fit(_X, _y)

    # END agscv_verbose ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    # good_tp must match good_sk_params
    @pytest.mark.parametrize('agscv_verbose,good_max_shifts,tpih,good_tp',
        (
            (True, None, True, 3),
            (False, 3, False, 3)
        )
    )
    def test_accepts_good_everything(
        self, AutoGridSearch, good_sk_logistic_params, _X, _y,
        agscv_verbose, good_max_shifts, tpih, good_tp,
    ):

        # sklearn_GSCV_kwargs, estimator, params, total_passes,
        # total_passes_is_hard, max_shifts, agscv_verbose

        # the parent GSCV wont check inputs until u try to fit
        _gscv = AutoGridSearch(
            estimator=sk_Logistic(),
            params=good_sk_logistic_params,
            total_passes=good_tp,
            total_passes_is_hard=tpih,
            max_shifts=good_max_shifts,
            agscv_verbose=agscv_verbose,
            scoring='accuracy',
            n_jobs=None,
            cv=5
        )

        _gscv.fit(_X, _y)


        assert _gscv.total_passes_is_hard is tpih
        if _gscv.total_passes_is_hard:
            assert _gscv.total_passes == good_tp
        elif not _gscv.total_passes_is_hard:
            assert _gscv.total_passes >= good_tp
        assert _gscv.max_shifts == good_max_shifts
        assert _gscv.agscv_verbose is agscv_verbose



class TestBoolInFixedIntegerFixedFloat:


    def test_bool_in_fixed_integer(self):

        AutoGridSearch = autogridsearch_wrapper(skl_GridSearchCV)

        X, y = make_classification(n_samples=50, n_features=5)

        _params = {
            'C': [np.logspace(-4, 4, 3), [3, 3, 3], 'soft_float'],
            'fit_intercept': [[True, False], [2, 1, 1], 'fixed_integer']
        }

        with pytest.raises(TypeError):
            _test_cls = AutoGridSearch(
                sk_Logistic(),
                _params,
                total_passes=3,
                total_passes_is_hard=True,
                max_shifts=2,
                agscv_verbose=False
            )

            _test_cls.fit(X, y)


    def test_bool_in_fixed_float(self):

        AutoGridSearch = autogridsearch_wrapper(skl_GridSearchCV)

        X, y = make_classification(n_samples=50, n_features=5)

        _params = {
            'C': [np.logspace(-4, 4, 3), [3, 3, 3], 'soft_float'],
            'fit_intercept': [[True, False], [2, 1, 1], 'fixed_float']
        }

        with pytest.raises(TypeError):
            _test_cls = AutoGridSearch(
                sk_Logistic(),
                _params,
                total_passes=3,
                total_passes_is_hard=True,
                max_shifts=2,
                agscv_verbose=False
            )

            _test_cls.fit(X, y)






