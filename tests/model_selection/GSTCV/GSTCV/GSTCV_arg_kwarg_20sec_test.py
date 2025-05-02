# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import re

import numpy as np

from sklearn.model_selection import KFold as sk_KFold
from pybear.model_selection.GSTCV._GSTCV.GSTCV import GSTCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    precision_score
)

from sklearn.preprocessing import OneHotEncoder as sk_OneHotEncoder

# wrap around RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV

from sklearn.linear_model import (
    LinearRegression as sk_LinearRegression,
    Ridge as sk_Ridge,
    RidgeClassifier as sk_RidgeClassifier, # wrap with CCCV
    LogisticRegression as sk_LogisticRegression,
    SGDClassifier as sk_SGDClassifier,
    SGDRegressor as sk_SGDRegressor
)


from dask_ml.linear_model import (
    LinearRegression as dask_LinearRegression,
    LogisticRegression as dask_LogisticRegression
)



class TestGSTCVInput:


    # def __init__(
    #     self,
    #     estimator,
    #     param_grid: ParamGridType,
    #     *,
    #     # thresholds can be a single number or list-type passed in
    #     # param_grid or applied universally via thresholds kwarg
    #     thresholds:
    #           Optional[Union[Iterable[Union[int, float]], int, float, None]]=None,
    #     scoring: Optional[ScorerInputType]='accuracy',
    #     n_jobs: Optional[Union[int,None]]=None,
    #     refit: Optional[RefitType]=True,
    #     cv: Optional[Union[int,None]]=None,
    #     verbose: Optional[Union[int, float, bool]]=0,
    #     pre_dispatch: Optional[Union[str, None]]='2*n_jobs',
    #     error_score: Optional[Union[Literal['raise'], int, float]]='raise',
    #     return_train_score: Optional[bool]=False
    # ):

    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @staticmethod
    @pytest.fixture(scope='function')
    def _GSTCV():
        return GSTCV(
            estimator=sk_LogisticRegression(C=1e-4, solver='saga', tol=1e-4),
            param_grid={},
            thresholds=[0.5],
            cv=2,
            scoring='accuracy',
            refit=False,
            return_train_score=False
        )

    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # estimator ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # must be an instance not the class! & be an estimator!

    @pytest.mark.parametrize('not_instantiated',
        (sk_OneHotEncoder, sk_LinearRegression, sk_Ridge, sk_RidgeClassifier,
        sk_LogisticRegression, sk_SGDClassifier, sk_SGDRegressor,
        CalibratedClassifierCV)
    )
    def test_estimator_rejects_not_instantiated(
        self, _GSTCV, not_instantiated, X_np, y_np
    ):

        with pytest.raises(
            TypeError,
            match=re.escape(
                "set_params() missing 1 required positional argument: 'self'"
            )
        ):
            _GSTCV.set_params(estimator=not_instantiated).fit(X_np, y_np)


    @pytest.mark.parametrize('non_estimator',
        (int, str, list, object, sk_OneHotEncoder)
    )
    def test_rejects_non_estimator(self, _GSTCV, non_estimator, X_np, y_np):

        with pytest.raises(AttributeError):
            _GSTCV.set_params(estimator=non_estimator()).fit(X_np, y_np)


    @pytest.mark.parametrize('non_classifier',
        (sk_LinearRegression, sk_Ridge, sk_SGDRegressor)
    )
    def test_estimator_rejects_non_classifier(
        self, _GSTCV, non_classifier, X_np, y_np
    ):

        with pytest.raises(AttributeError):
            _GSTCV.set_params(estimator=non_classifier()).fit(X_np, y_np)


    @pytest.mark.parametrize('good_classifiers',
        (sk_LogisticRegression, )
    )
    def test_estimator_accepts_non_dask_classifiers(
        self, _GSTCV, good_classifiers, X_np, y_np
    ):

        assert isinstance(
            _GSTCV.set_params(estimator=good_classifiers()).fit(X_np, y_np),
            type(_GSTCV)
        )


    @pytest.mark.parametrize('dask_non_classifiers',
        (dask_LinearRegression, )
    )
    def test_estimator_rejects_all_dask_non_classifiers(
        self, _GSTCV, dask_non_classifiers, X_np, y_np
    ):

        # must be an instance not the class! & be a classifier!
        with pytest.raises(AttributeError):
            _GSTCV.set_params(estimator=dask_non_classifiers()).fit(X_np, y_np)


    @pytest.mark.parametrize('dask_classifiers',
        (dask_LogisticRegression, )
    )
    def test_estimator_rejects_all_dask_classifiers(
        self, _GSTCV, dask_classifiers, X_np, y_np
    ):

        # must be an instance not the class! & be a classifier!
        with pytest.raises(TypeError):
            _GSTCV.set_params(estimator=dask_classifiers()).fit(X_np, y_np)

    # END estimator ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    # param_grid & thresholds ** * ** * ** * ** * ** * ** * ** * ** * **

    # param_grid: ParamGridType, dict[str, Union[list[any], npt.NDArray[any]]]
    # thresholds: Optional[Union[Iterable[Union[int, float]], int, float, None]]=None


    @staticmethod
    @pytest.fixture
    def good_threshes():
        return np.linspace(0, 1, 5)


    @staticmethod
    @pytest.fixture
    def good_param_grid():
        return [
            {'C': [1e-6, 1e-5, 1e-4], 'solver':['saga', 'lbfgs']},
            {'solver':['saga', 'lbfgs'], 'tol': [1e-4, 1e-6]},
            {'thresholds': [0.25], 'solver':['saga', 'lbfgs'], 'tol': [1e-4, 1e-5]}
        ]


    @pytest.mark.parametrize('empty_param_grid',
        ({}, [], [{}], [{}, {}])
    )
    def test_handling_of_empties(self, _GSTCV, empty_param_grid, X_np, y_np):

        assert isinstance(
            _GSTCV.set_params(param_grid=empty_param_grid).fit(X_np, y_np),
            type(_GSTCV)
        )


    def test_pg_thresh_accuracy_1(
        self, _GSTCV, good_threshes, good_param_grid, X_np, y_np
    ):

        # if param_grid had valid thresholds in it, it comes out the same as
        # it went in, regardless of passed threshes (dicts 1 & 3)

        out = _GSTCV.set_params(
            param_grid=good_param_grid[0],   # <==============
            thresholds=np.linspace(0,1,5)
        ).fit(X_np, y_np)

        assert isinstance(out, type(_GSTCV))

        _param_grid = out.get_params(deep=True)['param_grid']
        assert isinstance(_param_grid, dict)
        assert len(_param_grid) == 2
        assert np.array_equiv(_param_grid.keys(), good_param_grid[0].keys())
        for k, v in _param_grid.items():
            assert np.array_equiv(_param_grid[k], good_param_grid[0][k])


        out = _GSTCV.set_params(
            param_grid=good_param_grid[2],   # <==============
            thresholds=good_threshes
        ).fit(X_np, y_np)

        assert isinstance(out, type(_GSTCV))

        _param_grid = out.get_params(deep=True)['param_grid']
        assert isinstance(_param_grid, dict)
        assert len(_param_grid) == 3
        assert np.array_equiv(_param_grid.keys(), good_param_grid[2].keys())
        for k, v in _param_grid.items():
            assert np.array_equiv(_param_grid[k], good_param_grid[2][k])


    def test_pg_thresh_accuracy_2(self, _GSTCV, X_np, y_np):

        # if param_grid was not passed, but thresholds was, should be a param
        # grid with only the thresholds in it

        # notice testing pass as set
        out = _GSTCV.set_params(
            param_grid=None,
            thresholds={0, 0.25, 0.5, 0.75, 1}
        ).fit(X_np, y_np)

        assert isinstance(out, type(_GSTCV))

        _param_grid = out.get_params(deep=True)['param_grid']
        assert isinstance(_param_grid, type(None))

        # notice testing pass as list
        out = _GSTCV.set_params(
            param_grid={},
            thresholds=[0, 0.25, 0.5, 0.75, 1]
        ).fit(X_np, y_np)

        assert isinstance(out, type(_GSTCV))

        _param_grid = out.get_params(deep=True)['param_grid']
        assert isinstance(_param_grid, dict)
        assert len(_param_grid) == 0


    def test_pg_thresh_accuracy_3(self, _GSTCV, good_threshes, X_np, y_np):

        # if both param_grid and thresholds were not passed, should be one
        # param grid with default thresholds

        out = _GSTCV.set_params(
            param_grid=None,
            thresholds=None
        ).fit(X_np, y_np)

        assert isinstance(out, type(_GSTCV))

        _param_grid = out.get_params(deep=True)['param_grid']
        assert isinstance(_param_grid, type(None))


    def test_pg_thresh_accuracy_4(
        self, _GSTCV, good_param_grid, good_threshes, X_np, y_np
    ):

        # if param_grid was passed and did not have thresholds, should be the
        # same except have given thresholds in it. If thresholds was not
        # passed, default thresholds should be in it. (dict 2)

        # notice testing pass as set
        out = _GSTCV.set_params(
            param_grid=good_param_grid,
            thresholds={0, 0.25, 0.5, 0.75, 1}
        ).fit(X_np, y_np)

        assert isinstance(out, type(_GSTCV))

        _param_grid = out.get_params(deep=True)['param_grid']
        assert isinstance(_param_grid, list)
        assert len(_param_grid) == 3
        for _idx, _grid in enumerate(_param_grid):
            assert isinstance(_grid, dict)

            assert np.array_equiv(
                list(_grid.keys()),
                list(good_param_grid[_idx].keys())
            )
            for k,v in _grid.items():
                assert np.array_equiv(v, good_param_grid[_idx][k])

        # ** * ** *


        out = _GSTCV.set_params(
            param_grid=good_param_grid,
            thresholds=None
        ).fit(X_np, y_np)

        assert isinstance(out, type(_GSTCV))

        _param_grid = out.get_params(deep=True)['param_grid']
        assert isinstance(_param_grid, list)
        assert len(_param_grid) == 3
        for _idx, _grid in enumerate(_param_grid):
            assert isinstance(_grid, dict)

            assert np.array_equiv(
                list(_grid.keys()),
                list(good_param_grid[_idx].keys())
            )
            for k,v in _grid.items():
                assert np.array_equiv(v, good_param_grid[_idx][k])


    # END param_grid & thresholds ** * ** * ** * ** * ** * ** * ** * **


    # scoring ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # scoring: Optional[ScorerInputType]='accuracy'

    @pytest.mark.parametrize('junk_scoring', (0, 1, True, False, None, np.nan))
    def test_scoring_rejects_anything_not_str_callable_dict_iterable(
        self, _GSTCV, junk_scoring, X_np, y_np
    ):

        with pytest.raises(TypeError):
            _GSTCV.set_params(scoring=junk_scoring).fit(X_np, y_np)


    @pytest.mark.parametrize('junk_scoring',
        ('junk', 'garbage', 'trash', 'rubbish', 'waste', 'refuse')
    )
    def test_scoring_rejects_bad_strs(self, _GSTCV, junk_scoring, X_np, y_np):

        with pytest.raises(ValueError):
            _GSTCV.set_params(scoring=junk_scoring).fit(X_np, y_np)


    @pytest.mark.parametrize('good_scoring',
        ('accuracy', 'balanced_accuracy', 'precision', 'recall')
    )
    def test_scoring_accepts_good_strs(self, _GSTCV, good_scoring, X_np, y_np):

        assert isinstance(
            _GSTCV.set_params(scoring=good_scoring).fit(X_np, y_np),
            type(_GSTCV)
        )

        _scoring = _GSTCV.get_params(deep=True)['scoring']
        assert isinstance(_scoring, str)
        assert _scoring == good_scoring


    @pytest.mark.parametrize('junk_scoring',
        (lambda x: 'junk', lambda x: [0,1], lambda x,y: min, lambda x,y: x)
    )
    def test_scoring_rejects_non_num_callables(
        self, _GSTCV, junk_scoring, X_np, y_np
    ):

        with pytest.raises(ValueError):
            _GSTCV.set_params(scoring=junk_scoring).fit(X_np, y_np)


    def test_scoring_accepts_good_callable(self, _GSTCV, X_np, y_np):

        good_callable = lambda y1, y2: np.sum(np.array(y2)-np.array(y1))

        assert isinstance(
            _GSTCV.set_params(scoring=good_callable).fit(X_np, y_np),
            type(_GSTCV)
        )

        _scoring = _GSTCV.get_params(deep=True)['scoring']
        assert callable(_scoring)
        assert float(_scoring([1, 0, 1, 1], [1, 0, 0, 1]))


    @pytest.mark.parametrize('junk_scoring', ([], (), {}))
    def test_scoring_rejects_empty(self, _GSTCV, junk_scoring, X_np, y_np):

        with pytest.raises(ValueError):
            _GSTCV.set_params(scoring=junk_scoring).fit(X_np, y_np)


    @pytest.mark.parametrize('junk_lists',
        ([1,2,3], ('a','b','c'), {0,1,2}, ['trash', 'garbage', 'junk'])
    )
    def test_scoring_rejects_junk_lists(self, _GSTCV, junk_lists, X_np, y_np):

        with pytest.raises((TypeError, ValueError)):
            _GSTCV.set_params(scoring=junk_lists).fit(X_np, y_np)


    @pytest.mark.parametrize('good_lists',
        (['precision', 'recall'], ('accuracy','balanced_accuracy'),
         {'f1', 'balanced_accuracy', 'recall', 'precision'})
    )
    def test_scoring_accepts_good_lists(self, _GSTCV, good_lists, X_np, y_np):

        assert isinstance(
            _GSTCV.set_params(scoring=good_lists).fit(X_np, y_np),
            type(_GSTCV)
        )

        _scoring = _GSTCV.get_params(deep=True)['scoring']
        assert isinstance(_scoring, (list, tuple, set))
        assert np.array_equiv(sorted(_scoring), sorted(good_lists))


    @pytest.mark.parametrize('junk_dicts',
        ({'a':1, 'b':2}, {0:1, 1:2}, {0:[1,2,3], 1:[2,3,4]},
         {'metric1': lambda y1, y2: 'trash', 'metric2': lambda x: 1})
    )
    def test_scoring_rejects_junk_dicts(self, _GSTCV, junk_dicts, X_np, y_np):

        with pytest.raises(ValueError):
            _GSTCV.set_params(scoring=junk_dicts).fit(X_np, y_np)


    @pytest.mark.parametrize('good_dict',
        ({'accuracy': accuracy_score, 'f1': f1_score, 'recall': recall_score},
         {'metric1': precision_score, 'metric2': recall_score})
    )
    def test_scoring_accepts_good_dicts(self, _GSTCV, good_dict, X_np, y_np):

        assert isinstance(
            _GSTCV.set_params(scoring=good_dict).fit(X_np, y_np),
            type(_GSTCV)
        )

        _scoring = _GSTCV.get_params(deep=True)['scoring']
        assert isinstance(_scoring, dict)
        assert len(_scoring) == len(good_dict)
        for metric in good_dict:
            assert metric in _scoring
            assert callable(_scoring[metric])
            assert float(_scoring[metric]([0,1,0,1],[1,0,0,1]))


    # scoring ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # n_jobs ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # n_jobs: Optional[Union[int,None]]=None,

    @pytest.mark.parametrize('junk_njobs',
        (float('inf'), True, False, 'trash', min, [0,1], (0,1), {0,1}, {'a':1},
         lambda x: x)
    )
    def test_n_jobs_rejects_non_int_non_None(self, _GSTCV, junk_njobs, X_np, y_np):

        with pytest.raises(TypeError):
            _GSTCV.set_params(n_jobs=junk_njobs).fit(X_np, y_np)


    @pytest.mark.parametrize('bad_njobs', (-2, 0, 3.14))
    def test_n_jobs_rejects_bad_int(self, _GSTCV, bad_njobs, X_np, y_np):

        with pytest.raises(ValueError):
            _GSTCV.set_params(n_jobs=bad_njobs).fit(X_np, y_np)


    def test_n_jobs_None_returns_None(self, _GSTCV, X_np, y_np):

        assert isinstance(
            _GSTCV.set_params(n_jobs=None).fit(X_np, y_np),
            type(_GSTCV)
        )

        assert _GSTCV.get_params(deep=True)['n_jobs'] is None


    @pytest.mark.parametrize('good_njobs', (-1, 1, 5, 10))
    def test_n_jobs_otherwise_returns_given(self, _GSTCV, good_njobs, X_np, y_np):
        assert isinstance(
            _GSTCV.set_params(n_jobs=good_njobs).fit(X_np, y_np),
            type(_GSTCV)
        )

        assert _GSTCV.get_params(deep=True)['n_jobs'] == good_njobs


    # END n_jobs ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # refit ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # refit: Optional[RefitType]=True,


    one_scorer = {'accuracy': accuracy_score}


    two_scorers = {
        'accuracy': accuracy_score,
        'balanced_accuracy': balanced_accuracy_score
    }


    @pytest.mark.parametrize('n_scorers', (one_scorer, two_scorers))
    @pytest.mark.parametrize('junk_refit',
        (0, 1, 3.14, [0,1], (0,1), {0,1}, {'a':1})
    )
    def test_refit_rejects_junk(self, _GSTCV, n_scorers, junk_refit, X_np, y_np):
        with pytest.raises(TypeError):
            _GSTCV.set_params(refit=junk_refit, scoring=n_scorers).fit(X_np, y_np)


    @pytest.mark.parametrize('n_scorers', (one_scorer, two_scorers))
    @pytest.mark.parametrize('_callable',
        (lambda X: 0, lambda X: len(X['params']) - 1)
    )
    def test_refit_accepts_callable(self, _GSTCV, n_scorers, _callable, X_np, y_np):
        kwargs = {'refit': _callable, 'scoring': n_scorers}

        assert isinstance(
            _GSTCV.set_params(**kwargs).fit(X_np, y_np),
            type(_GSTCV)
        )
        assert _GSTCV.get_params(deep=True)['refit'] == _callable


    @pytest.mark.parametrize('n_scorers', (one_scorer, two_scorers))
    @pytest.mark.parametrize('_refit', (False, ))
    def test_refit_accepts_False(
        self, _GSTCV, n_scorers, _refit, X_np, y_np
    ):

        kwargs = {'refit': _refit, 'scoring': n_scorers}

        if len(n_scorers) == 1:
            assert isinstance(
                _GSTCV.set_params(**kwargs).fit(X_np, y_np),
                type(_GSTCV)
            )

        elif len(n_scorers) == 2:

            with pytest.warns():
                assert isinstance(
                    _GSTCV.set_params(**kwargs).fit(X_np, y_np),
                    type(_GSTCV)
                )

        assert _GSTCV.get_params(deep=True)['refit'] is False


    @pytest.mark.parametrize('n_scorers', (one_scorer,))
    def test_refit_single_accepts_true(self, _GSTCV, n_scorers, X_np, y_np):

        kwargs = {'refit': True, 'scoring': n_scorers}

        assert isinstance(
            _GSTCV.set_params(**kwargs).fit(X_np, y_np),
            type(_GSTCV)
        )

        assert _GSTCV.get_params(deep=True)['refit'] is True


    @pytest.mark.parametrize('n_scorers', (two_scorers,))
    def test_refit_multi_rejects_true(self, _GSTCV, n_scorers, X_np, y_np):

        kwargs = {'refit': True, 'scoring': n_scorers}

        with pytest.raises(ValueError):
            assert _GSTCV.set_params(**kwargs).fit(X_np, y_np)


    @pytest.mark.parametrize('n_scorers', (one_scorer, two_scorers))
    @pytest.mark.parametrize('junk_string', ('trash', 'garbage', 'junk'))
    def test_refit_rejects_junk_strings(
        self, _GSTCV, n_scorers, junk_string, X_np, y_np
    ):

        kwargs = {'refit': junk_string, 'scoring': n_scorers}

        with pytest.raises(ValueError):
            assert _GSTCV.set_params(**kwargs).fit(X_np, y_np)


    @pytest.mark.parametrize('n_scorers', (two_scorers,))
    def test_refit_accepts_good_strings(self, _GSTCV, n_scorers, X_np, y_np):

        data = (X_np, y_np)
        kwargs = lambda _scorer: {'refit': _scorer, 'scoring': n_scorers}
        
        if len(n_scorers) == 1:
            assert isinstance(
                _GSTCV.set_params(**kwargs('accuracy')).fit(*data),
                type(_GSTCV)
            )

            assert _GSTCV.get_params(deep=True)['refit'] == 'accuracy'

        if len(n_scorers) == 2:
            assert isinstance(
                _GSTCV.set_params(**kwargs('balanced_accuracy')).fit(*data),
                type(_GSTCV)
            )

            assert _GSTCV.get_params(deep=True)['refit'] == 'balanced_accuracy'

    # END refit ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # cv ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # cv: Optional[Union[int, Iterable, Generator, None]] = None

    @pytest.mark.parametrize('junk_cv',
        (2.718, 3.1416, True, False, 'trash', min, {'a': 1}, lambda x: x)
    )
    def test_cv_rejects_non_None_iter_int(self, _GSTCV, junk_cv, X_np, y_np):

        with pytest.raises(TypeError):
            _GSTCV.set_params(cv=junk_cv).fit(X_np, y_np)


    def test_cv_accepts_None(self, _GSTCV, X_np, y_np):

        assert isinstance(
            _GSTCV.set_params(cv=None).fit(X_np, y_np),
            type(_GSTCV)
        )
        assert _GSTCV.get_params(deep=True)['cv'] is None


    @pytest.mark.parametrize('bad_cv', (-1, 0, 1))
    def test_cv_value_error_less_than_2(self, _GSTCV, bad_cv, X_np, y_np):

        with pytest.raises(ValueError):
            _GSTCV.set_params(cv=bad_cv).fit(X_np, y_np)


    @pytest.mark.parametrize(f'good_int', (2, 3, 4, 5))
    def test_cv_accepts_good_int(self, _GSTCV, good_int, X_np, y_np):
        assert isinstance(
            _GSTCV.set_params(cv=good_int).fit(X_np, y_np),
            type(_GSTCV)
        )
        assert _GSTCV.get_params(deep=True)['cv'] == good_int


    @pytest.mark.parametrize(f'junk_iter', ([1, 2, 3], (True, False)))
    def test_cv_rejects_junk_iter_1(self, _GSTCV, junk_iter, X_np, y_np):

        with pytest.raises(TypeError):
            assert _GSTCV.set_params(cv=junk_iter).fit(X_np, y_np)


    @pytest.mark.parametrize(f'junk_iter',
        ([[1, 2, 3], [1, 2, 3], [2, 3, 4]], list('abcde'))
    )
    def test_cv_rejects_junk_iter_2(self, _GSTCV, junk_iter, X_np, y_np):

        with pytest.raises(ValueError):
            assert _GSTCV.set_params(
                cv=[[1, 2, 3], [1, 2, 3], [2, 3, 4]]
            ).fit(X_np, y_np)


    def test_cv_accepts_good_iter(self, _GSTCV, X_np, y_np):

        good_iter = sk_KFold(n_splits=3).split(X_np, y_np)

        _GSTCV.set_params(cv=good_iter).fit(X_np, y_np)


    def test_cv_rejects_empties(self, _GSTCV, X_np, y_np):

        with pytest.raises(ValueError):
            _GSTCV.set_params(cv=[()]).fit(X_np, y_np)

        with pytest.raises(ValueError):
            _GSTCV.set_params(cv=(_ for _ in range(0))).fit(X_np, y_np)

    # END cv ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # verbose ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @pytest.mark.parametrize('junk_verbose',
        (None, 'trash', [0, 1], (0, 1), {0, 1}, {'a': 1}, min, lambda x: x)
    )
    def test_verbose_rejects_non_num(self, _GSTCV, junk_verbose, X_np, y_np):
        # verbose: Optional[Union[int, float, bool]]=0,

        with pytest.raises(TypeError):
            _GSTCV.set_params(verbose=junk_verbose).fit(X_np, y_np)


    @pytest.mark.parametrize('bad_verbose', (-4, -3.14, -1))
    def test_verbose_rejects_negative(self, _GSTCV, bad_verbose, X_np, y_np):

        with pytest.raises(ValueError):
            _GSTCV.set_params(verbose=bad_verbose).fit(X_np, y_np)


    @pytest.mark.parametrize('good_verbose',(0, 1, 3.14, 1000))
    def test_verbose_accepts_any_pos_num(self, _GSTCV, good_verbose, X_np, y_np):
        # verbose: Optional[Union[int, float, bool]]=0,

        assert isinstance(
            _GSTCV.set_params(verbose=good_verbose).fit(X_np, y_np),
            type(_GSTCV)
        )

        assert _GSTCV.get_params(deep=True)['verbose'] == good_verbose


    # END verbose ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # error_score ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @ pytest.mark.parametrize('_junk_error_score',
        (None, True, False, [0,1], (0,1), {0,1}, {'a': 1}, min, lambda x: x)
    )
    def test_error_score_rejects_junk(
        self, _GSTCV, _junk_error_score, X_np, y_np
    ):
        # Optional[Union[Literal['raise'], int, float]] = 'raise'

        with pytest.raises(TypeError):
            _GSTCV.set_params(error_score=_junk_error_score).fit(X_np, y_np)


    def test_error_score_rejects_bad_str(self, _GSTCV, X_np, y_np):
        # Optional[Union[Literal['raise'], int, float]] = 'raise'

        with pytest.raises(ValueError):
            _GSTCV.set_params(error_score='garbage').fit(X_np, y_np)


    @pytest.mark.parametrize('good_error_score', (-1, 0, 1, 3.14, np.nan, 'raise'))
    def test_error_score_accepts_any_num_or_literal_raise(
        self, _GSTCV, good_error_score, X_np, y_np
    ):
        # Optional[Union[Literal['raise'], int, float]] = 'raise'

        assert isinstance(
            _GSTCV.set_params(error_score=good_error_score).fit(X_np, y_np),
            type(_GSTCV)
        )

        _error_score = _GSTCV.get_params(deep=True)['error_score']
        if _error_score is np.nan:
            assert good_error_score is np.nan
        else:
            assert _error_score == good_error_score

    # END error_score ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # return_train_score ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @ pytest.mark.parametrize('_junk_train_score',
        (0, 1, -1, 3.14, None, 'trash', [0,1], (0,1), {0,1}, {'a': 1},
         min, lambda x: x)
    )
    def test_train_score_rejects_non_bool(
        self, _GSTCV, _junk_train_score, X_np, y_np
    ):
        #return_train_score: Optional[bool]=False

        with pytest.raises(TypeError):
            _GSTCV.set_params(return_train_score=_junk_train_score).fit(X_np, y_np)


    @ pytest.mark.parametrize('good_train_score', (True, False))
    def test_train_score_accepts_bool(self, _GSTCV, good_train_score, X_np, y_np):

        #return_train_score: Optional[bool]=False

        assert isinstance(
            _GSTCV.set_params(return_train_score=good_train_score).fit(X_np, y_np),
            type(_GSTCV)
        )

        assert _GSTCV.get_params(deep=True)['return_train_score'] == \
               good_train_score

    # END return_train_score ** * ** * ** * ** * ** * ** * ** * ** * **












