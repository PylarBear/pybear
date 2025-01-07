# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#





from pybear.base.mixins._GetParamsMixin import GetParamsMixin

import numpy as np

import pytest

from pybear.model_selection import GSTCV
from sklearn.pipeline import Pipeline

pytest.skip(reason='pizza not started, not finished', allow_module_level=True)



"""
FROM benchmarking.GSCV_GSTCV__another_get_params_comparator

estimator is a pipeline with sk OneHotEncoder & sk LogisticRegression

deep = False
        skgscv_params        gstcv_params
0                  cv                  cv
1         error_score         error_score
2           estimator           estimator
3              n_jobs              n_jobs
4          param_grid          param_grid
5        pre_dispatch                   -
6               refit               refit
7  return_train_score  return_train_score          
8             scoring             scoring
9                   -          thresholds
10            verbose             verbose

deep = True
                               skgscv_params                              gstcv_params
0                                         cv                                        cv
1                                error_score                               error_score
2                          estimator__memory                         estimator__memory
3                           estimator__steps                          estimator__steps
4                         estimator__verbose                        estimator__verbose
5                          estimator__onehot                         estimator__onehot
6                        estimator__logistic                       estimator__logistic
7              estimator__onehot__categories             estimator__onehot__categories
8                    estimator__onehot__drop                   estimator__onehot__drop
9                   estimator__onehot__dtype                  estimator__onehot__dtype
10  estimator__onehot__feature_name_combiner  estimator__onehot__feature_name_combiner
11         estimator__onehot__handle_unknown         estimator__onehot__handle_unknown
12         estimator__onehot__max_categories         estimator__onehot__max_categories
13          estimator__onehot__min_frequency          estimator__onehot__min_frequency
14          estimator__onehot__sparse_output          estimator__onehot__sparse_output
15                    estimator__logistic__C                    estimator__logistic__C
16         estimator__logistic__class_weight         estimator__logistic__class_weight
17                 estimator__logistic__dual                 estimator__logistic__dual
18        estimator__logistic__fit_intercept        estimator__logistic__fit_intercept
19    estimator__logistic__intercept_scaling    estimator__logistic__intercept_scaling
20             estimator__logistic__l1_ratio             estimator__logistic__l1_ratio
21             estimator__logistic__max_iter             estimator__logistic__max_iter
22          estimator__logistic__multi_class          estimator__logistic__multi_class
23               estimator__logistic__n_jobs               estimator__logistic__n_jobs
24              estimator__logistic__penalty              estimator__logistic__penalty
25         estimator__logistic__random_state         estimator__logistic__random_state
26               estimator__logistic__solver               estimator__logistic__solver
27                  estimator__logistic__tol                  estimator__logistic__tol
28              estimator__logistic__verbose              estimator__logistic__verbose
29           estimator__logistic__warm_start           estimator__logistic__warm_start
30                                 estimator                                 estimator
31                                    n_jobs                                    n_jobs
32                                param_grid                                param_grid
33                              pre_dispatch                                         -
34                                     refit                                     refit
35                        return_train_score                        return_train_score
36                                   scoring                                   scoring
37                                         -                                thresholds
38                                   verbose                                   verbose


"""




# pizza things to test:
# builtin vars returns attrs in alphabetical order
# single estimator/transformer, wrapped in GSCV, wrapped in Pipe
# rejects non-bool deep
# attrs with leading or trailing underscore are removed
# probably shouldnt test against sklearn order that would require importing
# sklearn stuff, and we want to avoid that going forward
# we could use pybear GSTCV, and a mock estimator, but pipeline is unavoidable
# params should be the same before and after fit


class TestVarsReturnsAlphabetical(Fixtures):

    out = vars(DummyEstimator)

    assert isinstance(out, dict)

    assert np.array_equal(
        list(out.keys()),
        sorted(list(out.keys()))
    )


@pytest.mark.parametrize('top_level_object', ('single_est', 'GSCV', 'pipe'))
@pytest.mark.parametrize('state', ('prefit', 'postfit'))
class TestGetParams(Fixtures):

    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @staticmethod
    @pytest.fixture(scope='function')
    def DummyEstimator():

        class Foo(GetParamsMixin):

            def __init__(self):
                self._is_fitted = False
                self.dum_attr_ = 1
                self.apples = 4
                self.bananas = 7
                self.elephants = 3
                self.ethanol = 5
                self.fries = 9


            def reset(self):
                try:
                    delattr(self, '_random_fill')
                except:
                    pass

                self._is_fitted = False


            def fit(self, X, y=None):
                self.reset()
                self._random_fill = np.random.uniform(0, 1)
                self._is_fitted = True
                return self.partial_fit(X, y)


            def score(self, X, y=None):
                return np.random.uniform(0, 1)


            def transform(self, X):

                assert self._is_fitted

                return np.full(X.shape, self._random_fill)


        return Foo()  # <====== initialized


    @staticmethod
    @pytest.fixture(scope='function')
    def DummyGSTCV(DummyEstimator):

        return GSTCV(
            estimator=DummyEstimator,
            param_grid={
                'apples': [3, 4, 5],
                'bananas': [6, 7, 8],
                'thresholds': [0.5]
            }
        )


    @staticmethod
    @pytest.fixture(scope='function')
    def DummyPipe(DummyEstimator):

        return Pipeline(
            steps=[
                ('DummyEstimator', DummyEstimator)
            ],
            verbose=False
        )


    @staticmethod
    @pytest.fixture(scope='function')
    def _shape():
        return (10, 5)


    @staticmethod
    @pytest.fixture(scope='function')
    def _X_np(_shape):
        return np.random.randint(0, 10, _shape)


    @staticmethod
    @pytest.fixture(scope='function')
    def TopLevelObject(
        top_level_object, state, DummyEstimator, DummyGSTCV, DummyPipe, _X_np
    ):

        if top_level_object == 'single_est':
            foo = DummyEstimator()
        elif top_level_object == 'GSCV':
            foo = DummyGSTCV()
        elif top_level_object == 'pipe':
            foo = DummyPipe()
        else:
            raise Exception

        if state == 'postfit':
            foo.fit(_X_np)

        return foo


    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *



    @pytest.mark.parametrize('bad_deep',
        (0, 1, 3.14, None, 'trash', min, [0,1], (0,1), {'a':1}, lambda x: x)
    )
    def test_rejects_non_bool_deep(self, TopLevelObject, stabad_deep):

        with pytest.raises(ValueError):
            TopLevelObject.get_params(bad_deep)


    @pytest.mark.parametrize('bool_deep', (True, False))
    def test_accepts_bool_deep(
        self, DummyEstimator, DummyGSTCV, DummyPipe, bool_deep
    ):

        out = TopLevelObject.get_params(bool_deep)

        assert isinstance(out, dict)


    def test_single_estimator(
        self, state,
    ):

        # test shallow single estimator ** * ** * ** * ** * ** * ** * **

        # only test params' names, not values; GSTCVDask's defaults my be
        # different than daskGSCV's

        daskgscv_shallow = list(_dask_GSCV.get_params(deep=False).keys())
        gstcv_shallow = list(_dask_GSTCV.get_params(deep=False).keys())

        assert 'thresholds' in gstcv_shallow
        assert 'verbose' in gstcv_shallow
        # +2 for thresholds / verbose
        assert len(gstcv_shallow) == len(daskgscv_shallow) + 2

        gstcv_shallow.remove('thresholds')
        gstcv_shallow.remove('verbose')

        assert np.array_equiv(daskgscv_shallow, gstcv_shallow)

        # test shallow single estimator ** * ** * ** * ** * ** * ** * **

        # test deep single estimator ** * ** * ** * ** * ** * ** * ** *

        # only test params' names, not values; GSTCV's defaults my be
        # different than skGSCV's

        skgscv_deep = list(_dask_GSCV.get_params(deep=True).keys())
        gstcv_deep = list(_dask_GSTCV.get_params(deep=True).keys())

        assert 'thresholds' in gstcv_deep

        assert len(gstcv_deep) == len(skgscv_deep) + 2

        gstcv_deep.remove('thresholds')
        gstcv_deep.remove('verbose')

        assert np.array_equiv(skgscv_deep, gstcv_deep)

        # END test deep single estimator ** * ** * ** * ** * ** * ** * **


    @pytest.mark.parametrize('bad_deep',
        (0, 1, 3.14, None, 'trash', min, [0,1], (0,1), {'a':1}, lambda x: x)
    )
    def test_rejects_non_bool_deep(
        self, _refit, state, bad_deep,
        sk_GSTCV_est_log_one_scorer_prefit,
        sk_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_np,
        sk_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_np,
        sk_GSTCV_est_log_one_scorer_postfit_refit_fxn_fit_on_np,
        sk_GSTCV_pipe_log_one_scorer_prefit,
        sk_GSTCV_pipe_log_one_scorer_postfit_refit_false_fit_on_np,
        sk_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_np,
        sk_GSTCV_pipe_log_one_scorer_postfit_refit_fxn_fit_on_np
    ):

        if state == 'prefit':
            _GSTCV = sk_GSTCV_est_log_one_scorer_prefit
            _GSTCV_PIPE = sk_GSTCV_pipe_log_one_scorer_prefit
        elif state == 'postfit':
            if _refit is False:
                _GSTCV = sk_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_np
                _GSTCV_PIPE = sk_GSTCV_pipe_log_one_scorer_postfit_refit_false_fit_on_np
            elif _refit == 'accuracy':
                _GSTCV = sk_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_np
                _GSTCV_PIPE = sk_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_np
            elif callable(_refit):
                _GSTCV = sk_GSTCV_est_log_one_scorer_postfit_refit_fxn_fit_on_np
                _GSTCV_PIPE = sk_GSTCV_pipe_log_one_scorer_postfit_refit_fxn_fit_on_np


        with pytest.raises(ValueError):
            _GSTCV.get_params(bad_deep)

        with pytest.raises(ValueError):
            _GSTCV_PIPE.get_params(bad_deep)


    @pytest.mark.parametrize('state', ('prefit', 'postfit'))
    def test_no_pipe(
        self,
        state,
        _refit,
        sk_GSCV_est_log_one_scorer_prefit,
        sk_GSCV_est_log_one_scorer_postfit_refit_false,
        sk_GSCV_est_log_one_scorer_postfit_refit_str,
        sk_GSCV_est_log_one_scorer_postfit_refit_fxn,
        sk_GSTCV_est_log_one_scorer_prefit,
        sk_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_np,
        sk_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_np,
        sk_GSTCV_est_log_one_scorer_postfit_refit_fxn_fit_on_np
    ):

        # test shallow no pipe ** * ** * ** * ** * ** * ** * ** * ** *

        if state == 'prefit' and _refit is not False:
            pytest.skip(reason=f'redundant tests when in prefit state')

        if state == 'prefit':
            _GSCV = sk_GSCV_est_log_one_scorer_prefit
            _GSTCV = sk_GSTCV_est_log_one_scorer_prefit

        elif state == 'postfit':
            if _refit is False:
                _GSCV = sk_GSCV_est_log_one_scorer_postfit_refit_false
                _GSTCV = sk_GSTCV_est_log_one_scorer_postfit_refit_false_fit_on_np
            elif _refit == 'accuracy':
                _GSCV = sk_GSCV_est_log_one_scorer_postfit_refit_str
                _GSTCV = sk_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_np
            elif callable(_refit):
                _GSCV = sk_GSCV_est_log_one_scorer_postfit_refit_str
                _GSTCV = sk_GSTCV_est_log_one_scorer_postfit_refit_str_fit_on_np

        # only test params' names, not values; GSTCV's defaults may be
        # different than skGSCV's

        skgscv_shallow = list(_GSCV.get_params(deep=False).keys())
        gstcv_shallow = list(_GSTCV.get_params(deep=False).keys())

        assert 'thresholds' in gstcv_shallow

        assert len(gstcv_shallow) == len(skgscv_shallow)

        gstcv_shallow.remove('thresholds')
        skgscv_shallow.remove('pre_dispatch')

        assert np.array_equiv(skgscv_shallow, gstcv_shallow)

        # END test shallow no pipe ** * ** * ** * ** * ** * ** * ** * **

        # test deep no pipe ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # only test params' names, not values; GSTCV's defaults my be
        # different than skGSCV's

        skgscv_deep = list(_GSCV.get_params(deep=True).keys())
        gstcv_deep = list(_GSTCV.get_params(deep=True).keys())

        assert 'thresholds' in gstcv_deep

        assert len(gstcv_deep) == len(skgscv_deep)

        gstcv_deep.remove('thresholds')
        skgscv_deep.remove('pre_dispatch')

        assert np.array_equiv(skgscv_deep, gstcv_deep)

        # END test deep no pipe ** * ** * ** * ** * ** * ** * ** * ** *


    @pytest.mark.parametrize('_refit', (False, 'accuracy', lambda x: 0))
    @pytest.mark.parametrize('state', ('prefit', 'postfit'))
    def test_pipe(
        self, state, _refit,
        sk_GSCV_pipe_log_one_scorer_prefit,
        sk_GSCV_pipe_log_one_scorer_postfit_refit_false,
        sk_GSCV_pipe_log_one_scorer_postfit_refit_str,
        sk_GSCV_pipe_log_one_scorer_postfit_refit_fxn,
        sk_GSTCV_pipe_log_one_scorer_prefit,
        sk_GSTCV_pipe_log_one_scorer_postfit_refit_false_fit_on_np,
        sk_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_np,
        sk_GSTCV_pipe_log_one_scorer_postfit_refit_fxn_fit_on_np
    ):

        # test shallow pipe ** * ** * ** * ** * ** * ** * ** * ** * ** *

        if state == 'prefit' and _refit is not False:
            pytest.skip(reason=f'redundant tests when in prefit state')

        if state == 'prefit':
            _GSCV_PIPE = sk_GSCV_pipe_log_one_scorer_prefit
            _GSTCV_PIPE = sk_GSTCV_pipe_log_one_scorer_prefit

        elif state == 'postfit':
            if _refit is False:
                _GSCV_PIPE = sk_GSCV_pipe_log_one_scorer_postfit_refit_false
                _GSTCV_PIPE = sk_GSTCV_pipe_log_one_scorer_postfit_refit_false_fit_on_np
            elif _refit == 'accuracy':
                _GSCV_PIPE = sk_GSCV_pipe_log_one_scorer_postfit_refit_str
                _GSTCV_PIPE = sk_GSTCV_pipe_log_one_scorer_postfit_refit_str_fit_on_np
            elif callable(_refit):
                _GSCV_PIPE = sk_GSCV_pipe_log_one_scorer_postfit_refit_fxn
                _GSTCV_PIPE = sk_GSTCV_pipe_log_one_scorer_postfit_refit_fxn_fit_on_np


        assert _GSCV_PIPE.estimator.steps[0][0] == 'sk_StandardScaler'
        assert _GSCV_PIPE.estimator.steps[1][0] == 'sk_logistic'

        assert _GSTCV_PIPE.estimator.steps[0][0] == 'sk_StandardScaler'
        assert _GSTCV_PIPE.estimator.steps[1][0] == 'sk_logistic'

        # only test params' names, not values; GSTCV's defaults my be
        # different than skGSCV's

        skgscv_shallow = list(_GSCV_PIPE.get_params(deep=False).keys())
        gstcv_shallow = list(_GSTCV_PIPE.get_params(deep=False).keys())

        assert 'thresholds' in gstcv_shallow

        # +1 for thresholds
        assert len(gstcv_shallow) == len(skgscv_shallow)

        assert len(skgscv_shallow) == 10
        assert len(gstcv_shallow) == 10

        gstcv_shallow.remove('thresholds')
        skgscv_shallow.remove('pre_dispatch')

        assert np.array_equiv(skgscv_shallow, gstcv_shallow)

        # END test shallow pipe ** * ** * ** * ** * ** * ** * ** * ** *


        # test deep pipe ** * ** * ** * ** * ** * ** * ** * ** * ** * **


        assert _GSCV_PIPE.estimator.steps[0][0] == 'sk_StandardScaler'
        assert _GSCV_PIPE.estimator.steps[1][0] == 'sk_logistic'

        assert _GSTCV_PIPE.estimator.steps[0][0] == 'sk_StandardScaler'
        assert _GSTCV_PIPE.estimator.steps[1][0] == 'sk_logistic'

        # only test params' names, not values; GSTCV's defaults my be
        # different than skGSCV's

        skgscv_deep = list(_GSCV_PIPE.get_params(deep=True).keys())
        gstcv_deep = list(_GSTCV_PIPE.get_params(deep=True).keys())

        assert 'thresholds' in gstcv_deep


        assert len(gstcv_deep) == len(skgscv_deep)

        assert len(skgscv_deep) == 33
        assert len(gstcv_deep) == 33

        gstcv_deep.remove('thresholds')
        skgscv_deep.remove('pre_dispatch')

        assert np.array_equiv(skgscv_deep, gstcv_deep)

        # END test deep pipe ** * ** * ** * ** * ** * ** * ** * ** * ** *



























