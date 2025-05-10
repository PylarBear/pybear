# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np


from sklearn.preprocessing import OneHotEncoder as sk_OneHotEncoder
from sklearn.model_selection import KFold as sk_KFold

# wrap around RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV

from sklearn.linear_model import (
    LinearRegression as sk_LinearRegression,
    Ridge as sk_Ridge,
    RidgeClassifier as sk_RidgeClassifier,  # wrap with CCCV
    LogisticRegression as sk_LogisticRegression,
    SGDClassifier as sk_SGDClassifier
)

from dask_ml.linear_model import (
    LinearRegression as dask_LinearRegression,
    LogisticRegression as dask_LogisticRegression
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

from pybear.model_selection.GSTCV._GSTCV.GSTCV import GSTCV



class TestInitValidation:

    # pizza update this or delete it
    #     def __init__(
    #         self,
    #         estimator: ClassifierProtocol,
    #         param_grid: Union[dict[str, Sequence[Any]], Sequence[dict[str, Sequence[Any]]]],
    #         *,
    #         thresholds: Optional[Union[None, numbers.Real, Sequence[numbers.Real]]]=None,
    #         scoring: Optional[
    #             Union[list[str], dict[str, Callable], str, Callable]
    #         ]='accuracy',
    #         n_jobs: Optional[Union[numbers.Integral, None]]=None,
    #         refit: Optional[Union[bool, str, Callable]]=True,
    #         cv: Optional[Union[numbers.Integral, Iterable, None]]=None,
    #         verbose: Optional[numbers.Real]=0,
    #         pre_dispatch: Optional[
    #             Union[Literal['all'], str, numbers.Integral]
    #         ]='2*n_jobs',
    #         error_score: Optional[Union[Literal['raise'], numbers.Real]]='raise',
    #         return_train_score: Optional[bool]=False
    #     ) -> None:






    # fixtures ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    # pizza there are a lot of WIP scorer dicts in here see if can streamline

    @staticmethod
    @pytest.fixture(scope='function')
    def base_gstcv(
        sk_est_log, param_grid_sk_log, standard_cv_int,
        standard_error_score, standard_WIP_scorer
    ):
        # dont overwrite a session fixture with new params!

        return GSTCV(
            estimator=sk_est_log,
            param_grid=param_grid_sk_log,
            thresholds=np.linspace(0,1,11),
            cv=standard_cv_int,
            error_score=standard_error_score,
            verbose=10,
            scoring=standard_WIP_scorer,
            refit=False,
            n_jobs=-1,
            pre_dispatch='2*n_jobs',
            return_train_score=True
        )



    # END fixtures ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **


    # estimator v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^

    # must be an instance not the class! & be a classifier!

    @pytest.mark.parametrize('junk_estimator',
        (-1, 0, 1, 3.14, True, False, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_estimator(
        self, X_np, y_np, junk_estimator, base_gstcv
    ):
        # dont use set_params here
        base_gstcv.estimator=junk_estimator

        with pytest.raises(AttributeError):
            base_gstcv.fit(X_np, y_np)


    @pytest.mark.parametrize('non_estimator',
        (int, str, list, object, sk_OneHotEncoder)
    )
    def test_rejects_non_estimator(
        self, base_gstcv, non_estimator, X_np, y_np
    ):

        base_gstcv.estimator=non_estimator()

        with pytest.raises(AttributeError):
            base_gstcv.fit(X_np, y_np)


    @pytest.mark.parametrize('non_classifier',
        (sk_LinearRegression(), sk_Ridge(), sk_RidgeClassifier())
    )
    def test_rejects_non_classifier(
        self, X_np, y_np, non_classifier, base_gstcv
    ):

        base_gstcv.set_params(estimator=non_classifier)

        with pytest.raises(AttributeError):
            base_gstcv.fit(X_np, y_np)


    @pytest.mark.parametrize('not_instantiated',
        (sk_LogisticRegression, sk_SGDClassifier, CalibratedClassifierCV)
    )
    def test_rejects_classifier_not_instantiated(
        self, base_gstcv, not_instantiated, X_np, y_np
    ):

        with pytest.raises(TypeError):
            base_gstcv.set_params(estimator=not_instantiated)


    def test_estimator_rejects_dask_non_classifier(self, base_gstcv, X_np, y_np):

        base_gstcv.set_params(estimator=dask_LinearRegression())

        with pytest.raises(AttributeError):
            base_gstcv.fit(X_np, y_np)


    def test_estimator_rejects_dask_classifier(self, base_gstcv, X_np, y_np):

        base_gstcv.set_params(estimator=dask_LogisticRegression())

        with pytest.raises(TypeError):
            base_gstcv.fit(X_np, y_np)


    def test_estimator_accepts_non_dask_classifiers(self, base_gstcv, X_np, y_np):

        base_gstcv.set_params(estimator=sk_LogisticRegression())

        assert isinstance(base_gstcv.fit(X_np, y_np), type(base_gstcv))

    # END estimator v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^


    # param_grid v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
    @pytest.mark.parametrize('junk_param_grid',
        (-1, 0, 1, 3.14, True, False, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         lambda x: x)
    )
    def test_rejects_junk_param_grid(
        self, X_np, y_np, base_gstcv, junk_param_grid
    ):

        base_gstcv.set_params(param_grid=junk_param_grid)

        with pytest.raises(TypeError):
            base_gstcv.fit(X_np, y_np)


    @pytest.mark.parametrize('empty_p_g', ({}, [], [{}], [{}, {}]))
    def test_handling_of_empties(
        self, base_gstcv, empty_p_g, X_np, y_np
    ):

        base_gstcv.set_params(param_grid=empty_p_g)

        assert isinstance(base_gstcv.fit(X_np, y_np), type(base_gstcv))

    # END param_grid v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v


    # thresholds v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v
    @pytest.mark.parametrize('junk_thresholds',
        (-1, 3.14, True, False, 'trash', min, ['a', 'b'], ('a', 'b'),
         {'a', 'b'}, lambda x: x)
    )
    def test_rejects_junk_thresholds(
        self, X_np, y_np, base_gstcv, junk_thresholds
    ):

        base_gstcv.set_params(thresholds=junk_thresholds)

        with pytest.raises((TypeError, ValueError)):
            base_gstcv.fit(X_np, y_np)


    @pytest.mark.parametrize('bad_thresholds', ({'a': 1}, {0: 1}, {0: 'b'}))
    def test_rejects_bad_thresholds(
        self, X_np, y_np, base_gstcv, bad_thresholds
    ):

        base_gstcv.set_params(thresholds=bad_thresholds)

        with pytest.raises(TypeError):
            base_gstcv.fit(X_np, y_np)

    # END thresholds v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v


    # joint param_grid & thresholds v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^

    # param_grid: dict[str, list[Any]] or list[dict[str, list[Any]]]
    # thresholds: Optional[Union[Sequence[numbers.Real], numbers.Real, None]]=None


    @staticmethod
    @pytest.fixture
    def good_param_grid():
        return [
            {'C': [1e-6, 1e-5, 1e-4], 'solver':['saga', 'lbfgs']},
            {'solver':['saga', 'lbfgs'], 'tol': [1e-4, 1e-6]},
            {'thresholds': [0.25], 'solver':['saga', 'lbfgs'], 'tol': [1e-4, 1e-5]}
        ]


    def test_pg_thresh_accuracy_1(
        self, base_gstcv, standard_thresholds, good_param_grid, X_np, y_np
    ):

        # if param_grid had valid thresholds in it, it comes out the same as
        # it went in, regardless of passed threshes (dicts 1 & 3)

        out = base_gstcv.set_params(
            param_grid=good_param_grid[0],   # <==============
            thresholds=np.linspace(0,1,5)
        ).fit(X_np, y_np)

        assert isinstance(out, type(base_gstcv))

        _param_grid = out.get_params(deep=True)['param_grid']
        assert isinstance(_param_grid, dict)
        assert len(_param_grid) == 2
        assert np.array_equiv(_param_grid.keys(), good_param_grid[0].keys())
        for k, v in _param_grid.items():
            assert np.array_equiv(_param_grid[k], good_param_grid[0][k])


        out = base_gstcv.set_params(
            param_grid=good_param_grid[2],   # <==============
            thresholds=standard_thresholds
        ).fit(X_np, y_np)

        assert isinstance(out, type(base_gstcv))

        _param_grid = out.get_params(deep=True)['param_grid']
        assert isinstance(_param_grid, dict)
        assert len(_param_grid) == 3
        assert np.array_equiv(_param_grid.keys(), good_param_grid[2].keys())
        for k, v in _param_grid.items():
            assert np.array_equiv(_param_grid[k], good_param_grid[2][k])


    def test_pg_thresh_accuracy_2(self, base_gstcv, X_np, y_np):

        # if param_grid was not passed, but thresholds was, should be a param
        # grid with only the thresholds in it

        # notice testing pass as set
        out = base_gstcv.set_params(
            param_grid=[],
            thresholds={0, 0.25, 0.5, 0.75, 1}
        ).fit(X_np, y_np)

        assert isinstance(out, type(base_gstcv))

        _param_grid = out.get_params(deep=True)['param_grid']
        assert isinstance(_param_grid, list)

        # notice testing pass as list
        out = base_gstcv.set_params(
            param_grid={},
            thresholds=[0, 0.25, 0.5, 0.75, 1]
        ).fit(X_np, y_np)

        assert isinstance(out, type(base_gstcv))

        _param_grid = out.get_params(deep=True)['param_grid']
        assert isinstance(_param_grid, dict)
        assert len(_param_grid) == 0


    def test_pg_thresh_accuracy_3(
        self, base_gstcv, standard_thresholds, X_np, y_np
    ):

        # if both param_grid and thresholds were not passed, should be one
        # param grid with default thresholds

        out = base_gstcv.set_params(
            param_grid=[],
            thresholds=None
        ).fit(X_np, y_np)

        assert isinstance(out, type(base_gstcv))

        _param_grid = out.get_params(deep=True)['param_grid']
        assert isinstance(_param_grid, list)


    def test_pg_thresh_accuracy_4(
        self, base_gstcv, good_param_grid, standard_thresholds, X_np, y_np
    ):

        # if param_grid was passed and did not have thresholds, should be the
        # same except have given thresholds in it. If thresholds was not
        # passed, default thresholds should be in it. (dict 2)

        # notice testing pass as set
        out = base_gstcv.set_params(
            param_grid=good_param_grid,
            thresholds={0, 0.25, 0.5, 0.75, 1}
        ).fit(X_np, y_np)

        assert isinstance(out, type(base_gstcv))

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


        out = base_gstcv.set_params(
            param_grid=good_param_grid,
            thresholds=None
        ).fit(X_np, y_np)

        assert isinstance(out, type(base_gstcv))

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

    # END joint param_grid & thresholds v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^


    # cv v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v
    # cv: Optional[Union[int, Iterable, Generator, None]] = None

    @pytest.mark.parametrize('junk_cv',
        (-1, 0, 1, 3.14, [0, 1], (0, 1), {0, 1}, True, False, 'trash', min,
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_cv(
        self, X_np, y_np, base_gstcv, junk_cv
    ):

        base_gstcv.set_params(cv=junk_cv)

        with pytest.raises((ValueError, TypeError)):
            base_gstcv.fit(X_np, y_np)


    def test_cv_accepts_None(self, base_gstcv, X_np, y_np):

        assert isinstance(
            base_gstcv.set_params(cv=None).fit(X_np, y_np),
            type(base_gstcv)
        )
        assert base_gstcv.get_params(deep=True)['cv'] is None


    @pytest.mark.parametrize('bad_cv', (-1, 0, 1))
    def test_cv_value_error_less_than_2(
        self, base_gstcv, bad_cv, X_np, y_np
    ):

        with pytest.raises(ValueError):
            base_gstcv.set_params(cv=bad_cv).fit(X_np, y_np)


    @pytest.mark.parametrize(f'good_int', (2, 3, 4, 5))
    def test_cv_accepts_good_int(
        self, base_gstcv, good_int, X_np, y_np
    ):
        assert isinstance(
            base_gstcv.set_params(cv=good_int).fit(X_np, y_np),
            type(base_gstcv)
        )
        assert base_gstcv.get_params(deep=True)['cv'] == good_int


    @pytest.mark.parametrize(f'junk_iter', ([1, 2, 3], (True, False)))
    def test_cv_rejects_junk_iter_1(
        self, base_gstcv, junk_iter, X_np, y_np
    ):

        with pytest.raises(TypeError):
            assert base_gstcv.set_params(cv=junk_iter).fit(X_np, y_np)


    @pytest.mark.parametrize(f'junk_iter',
        ([[1, 2, 3], [1, 2, 3], [2, 3, 4]], list('abcde'))
    )
    def test_cv_rejects_junk_iter_2(
        self, base_gstcv, junk_iter, X_np, y_np
    ):

        with pytest.raises(ValueError):
            assert base_gstcv.set_params(
                cv=[[1, 2, 3], [1, 2, 3], [2, 3, 4]]
            ).fit(X_np, y_np)


    def test_cv_accepts_good_iter(self, base_gstcv, X_np, y_np):

        good_iter = sk_KFold(n_splits=3).split(X_np, y_np)

        base_gstcv.set_params(cv=good_iter).fit(X_np, y_np)


    def test_cv_rejects_empties(self, base_gstcv, X_np, y_np):

        with pytest.raises(ValueError):
            base_gstcv.set_params(cv=[()]).fit(X_np, y_np)

        with pytest.raises(ValueError):
            base_gstcv.set_params(cv=(_ for _ in range(0))).fit(X_np, y_np)

    # END cv v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v


    # error_score v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
    # Optional[Union[Literal['raise'], numbers.Real]] = 'raise'

    @pytest.mark.parametrize('junk_error_score',
        (True, False, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_error_score(
        self, X_np, y_np, base_gstcv, junk_error_score
    ):

        base_gstcv.set_params(error_score=junk_error_score)

        with pytest.raises((TypeError, ValueError)):
            base_gstcv.fit(X_np, y_np)


    def test_error_score_rejects_bad_str(
        self, base_gstcv, X_np, y_np
    ):

        with pytest.raises(ValueError):
            base_gstcv.set_params(error_score='garbage').fit(X_np, y_np)


    @pytest.mark.parametrize('good_error_score', (-1, 0, 1, 3.14, np.nan, 'raise'))
    def test_error_score_accepts_any_num_or_literal_raise(
        self, base_gstcv, good_error_score, X_np, y_np
    ):

        base_gstcv.set_params(error_score=good_error_score)
        assert isinstance(base_gstcv.fit(X_np, y_np), type(base_gstcv))

        _error_score = base_gstcv.get_params(deep=True)['error_score']
        if _error_score is np.nan:
            assert good_error_score is np.nan
        else:
            assert _error_score == good_error_score

    # END error_score v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^


    # verbose v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
    # verbose: Optional[numbers.Real]=0

    @pytest.mark.parametrize('junk_verbose',
        (-10, -1, None, 'trash', min, [0, 1], (0, 1), {0, 1}, {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_verbose(
        self, X_np, y_np, base_gstcv, junk_verbose
    ):

        base_gstcv.set_params(verbose=junk_verbose)

        with pytest.raises((TypeError, ValueError)):
            base_gstcv.fit(X_np, y_np)


    @pytest.mark.parametrize('junk_verbose',
        (None, 'trash', [0, 1], (0, 1), {0, 1}, {'a': 1}, min, lambda x: x)
    )
    def test_verbose_rejects_non_num(
        self, base_gstcv, junk_verbose, X_np, y_np
    ):

        with pytest.raises(TypeError):
            base_gstcv.set_params(verbose=junk_verbose).fit(X_np, y_np)


    @pytest.mark.parametrize('bad_verbose', (-4, -3.14, -1))
    def test_verbose_rejects_negative(
        self, base_gstcv, bad_verbose, X_np, y_np
    ):

        with pytest.raises(ValueError):
            base_gstcv.set_params(verbose=bad_verbose).fit(X_np, y_np)


    @pytest.mark.parametrize('good_verbose',(0, 1, 3.14, 1000))
    def test_verbose_accepts_any_pos_num(
        self, base_gstcv, good_verbose, X_np, y_np
    ):

        assert isinstance(
            base_gstcv.set_params(verbose=good_verbose).fit(X_np, y_np),
            type(base_gstcv)
        )

        assert base_gstcv.get_params(deep=True)['verbose'] == good_verbose

    # END verbose v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^


    # refit v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
    # refit: Optional[RefitType]=True


    one_scorer = {'accuracy': accuracy_score}

    two_scorers = {
        'accuracy': accuracy_score,
        'balanced_accuracy': balanced_accuracy_score
    }

    @pytest.mark.parametrize('n_scorers', (one_scorer, two_scorers))
    @pytest.mark.parametrize('junk_refit',
        (0, 1, 3.14, [0, 1], (0, 1), {0, 1}, {'a': 1})
    )
    def test_refit_rejects_junk(
        self, base_gstcv, n_scorers, junk_refit, X_np, y_np
    ):

        base_gstcv.set_params(refit=junk_refit, scoring=n_scorers)

        with pytest.raises(TypeError):
            base_gstcv.fit(X_np, y_np)


    @pytest.mark.parametrize('n_scorers', (one_scorer, two_scorers))
    @pytest.mark.parametrize('_callable',
        (lambda X: 0, lambda X: len(X['params']) - 1)
    )
    def test_refit_accepts_callable(
        self, base_gstcv, n_scorers, _callable, X_np, y_np
    ):

        kwargs = {'refit': _callable, 'scoring': n_scorers}

        assert isinstance(
            base_gstcv.set_params(**kwargs).fit(X_np, y_np),
            type(base_gstcv)
        )
        assert base_gstcv.get_params(deep=True)['refit'] == _callable


    @pytest.mark.parametrize('n_scorers', (one_scorer, two_scorers))
    @pytest.mark.parametrize('_refit', (False,))
    def test_refit_accepts_False(
        self, base_gstcv, n_scorers, _refit, X_np, y_np
    ):

        kwargs = {'refit': _refit, 'scoring': n_scorers}

        if len(n_scorers) == 1:
            assert isinstance(
                base_gstcv.set_params(**kwargs).fit(X_np, y_np),
                type(base_gstcv)
            )

        elif len(n_scorers) == 2:

            with pytest.warns():
                assert isinstance(
                    base_gstcv.set_params(**kwargs).fit(X_np, y_np),
                    type(base_gstcv)
                )

        assert base_gstcv.get_params(deep=True)['refit'] is False


    @pytest.mark.parametrize('n_scorers', (one_scorer,))
    def test_refit_single_accepts_true(
        self, base_gstcv, n_scorers, X_np, y_np
    ):

        kwargs = {'refit': True, 'scoring': n_scorers}

        assert isinstance(
            base_gstcv.set_params(**kwargs).fit(X_np, y_np),
            type(base_gstcv)
        )

        assert base_gstcv.get_params(deep=True)['refit'] is True


    @pytest.mark.parametrize('n_scorers', (two_scorers,))
    def test_refit_multi_rejects_true(
        self, base_gstcv, n_scorers, X_np, y_np
    ):

        kwargs = {'refit': True, 'scoring': n_scorers}

        with pytest.raises(ValueError):
            assert base_gstcv.set_params(**kwargs).fit(X_np, y_np)


    @pytest.mark.parametrize('n_scorers', (one_scorer, two_scorers))
    @pytest.mark.parametrize('junk_string', ('trash', 'garbage', 'junk'))
    def test_refit_rejects_junk_strings(
        self, base_gstcv, n_scorers, junk_string, X_np, y_np
    ):

        kwargs = {'refit': junk_string, 'scoring': n_scorers}

        with pytest.raises(ValueError):
            assert base_gstcv.set_params(**kwargs).fit(X_np, y_np)


    @pytest.mark.parametrize('n_scorers', (two_scorers,))
    def test_refit_accepts_good_strings(
        self, base_gstcv, n_scorers, X_np, y_np
    ):

        data = (X_np, y_np)
        kwargs = lambda _scorer: {'refit': _scorer, 'scoring': n_scorers}

        if len(n_scorers) == 1:
            assert isinstance(
                base_gstcv.set_params(**kwargs('accuracy')).fit(*data),
                type(base_gstcv)
            )

            assert base_gstcv.get_params(deep=True)['refit'] == 'accuracy'

        if len(n_scorers) == 2:
            base_gstcv.set_params(**kwargs('balanced_accuracy'))
            assert isinstance(base_gstcv.fit(*data), type(base_gstcv))
            assert base_gstcv.get_params(deep=True)['refit'] == \
                   'balanced_accuracy'

    # END refit v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^


    # scoring v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
    # scoring: Optional[ScorerInputType]='accuracy'

    @pytest.mark.parametrize('junk_scoring',
        (-1, 0, 1, 3.14, True, False, None, 'trash', min, [0, 1], (0, 1),
         {'a': 1}, {0: 1}, {'trash': 'junk'}, lambda x: x)
    )
    def test_rejects_junk_scoring(
        self, X_np, y_np, base_gstcv, junk_scoring
    ):

        base_gstcv.set_params(scoring=junk_scoring)

        with pytest.raises((TypeError, ValueError)):
            base_gstcv.fit(X_np, y_np)


    @pytest.mark.parametrize('junk_scoring', (0, 1, True, False, None, np.nan))
    def test_scoring_rejects_anything_not_str_callable_dict_iterable(
        self, base_gstcv, junk_scoring, X_np, y_np
    ):

        with pytest.raises(TypeError):
            base_gstcv.set_params(scoring=junk_scoring).fit(X_np, y_np)


    @pytest.mark.parametrize('junk_scoring',
        ('junk', 'garbage', 'trash', 'rubbish', 'waste', 'refuse')
    )
    def test_scoring_rejects_bad_strs(
        self, base_gstcv, junk_scoring, X_np, y_np
    ):

        with pytest.raises(ValueError):
            base_gstcv.set_params(scoring=junk_scoring).fit(X_np, y_np)


    @pytest.mark.parametrize('good_scoring',
        ('accuracy', 'balanced_accuracy', 'precision', 'recall')
    )
    def test_scoring_accepts_good_strs(
        self, base_gstcv, good_scoring, X_np, y_np
    ):

        base_gstcv.set_params(scoring=good_scoring)
        assert isinstance(base_gstcv.fit(X_np, y_np), type(base_gstcv))

        _scoring = base_gstcv.get_params(deep=True)['scoring']
        assert isinstance(_scoring, str)
        assert _scoring == good_scoring


    @pytest.mark.parametrize('junk_scoring',
        (lambda x: 'junk', lambda x: [0,1], lambda x,y: min, lambda x,y: x)
    )
    def test_scoring_rejects_non_num_callables(
        self, base_gstcv, junk_scoring, X_np, y_np
    ):

        with pytest.raises(ValueError):
            base_gstcv.set_params(scoring=junk_scoring).fit(X_np, y_np)


    def test_scoring_accepts_good_callable(
        self, base_gstcv, X_np, y_np
    ):

        good_callable = lambda y1, y2: np.sum(np.array(y2)-np.array(y1))

        base_gstcv.set_params(scoring=good_callable)
        assert isinstance(base_gstcv.fit(X_np, y_np), type(base_gstcv))

        _scoring = base_gstcv.get_params(deep=True)['scoring']
        assert callable(_scoring)
        assert float(_scoring([1, 0, 1, 1], [1, 0, 0, 1]))


    @pytest.mark.parametrize('junk_scoring', ([], (), {}))
    def test_scoring_rejects_empty(
        self, base_gstcv, junk_scoring, X_np, y_np
    ):

        with pytest.raises(ValueError):
            base_gstcv.set_params(scoring=junk_scoring).fit(X_np, y_np)


    @pytest.mark.parametrize('junk_lists',
        ([1,2,3], ('a','b','c'), {0,1,2}, ['trash', 'garbage', 'junk'])
    )
    def test_scoring_rejects_junk_lists(
        self, base_gstcv, junk_lists, X_np, y_np
    ):

        with pytest.raises((TypeError, ValueError)):
            base_gstcv.set_params(scoring=junk_lists).fit(X_np, y_np)


    @pytest.mark.parametrize('good_lists',
        (['precision', 'recall'], ('accuracy','balanced_accuracy'),
         {'f1', 'balanced_accuracy', 'recall', 'precision'})
    )
    def test_scoring_accepts_good_lists(
        self, base_gstcv, good_lists, X_np, y_np
    ):

        assert isinstance(
            base_gstcv.set_params(scoring=good_lists).fit(X_np, y_np),
            type(base_gstcv)
        )

        _scoring = base_gstcv.get_params(deep=True)['scoring']
        assert isinstance(_scoring, (list, tuple, set))
        assert np.array_equiv(sorted(_scoring), sorted(good_lists))


    @pytest.mark.parametrize('junk_dicts',
        ({'a':1, 'b':2}, {0:1, 1:2}, {0:[1,2,3], 1:[2,3,4]},
         {'metric1': lambda y1, y2: 'trash', 'metric2': lambda x: 1})
    )
    def test_scoring_rejects_junk_dicts(
        self, base_gstcv, junk_dicts, X_np, y_np
    ):

        with pytest.raises(ValueError):
            base_gstcv.set_params(scoring=junk_dicts).fit(X_np, y_np)


    @pytest.mark.parametrize('good_dict',
        ({'accuracy': accuracy_score, 'f1': f1_score, 'recall': recall_score},
         {'metric1': precision_score, 'metric2': recall_score})
    )
    def test_scoring_accepts_good_dicts(
        self, base_gstcv, good_dict, X_np, y_np
    ):

        assert isinstance(
            base_gstcv.set_params(scoring=good_dict).fit(X_np, y_np),
            type(base_gstcv)
        )

        _scoring = base_gstcv.get_params(deep=True)['scoring']
        assert isinstance(_scoring, dict)
        assert len(_scoring) == len(good_dict)
        for metric in good_dict:
            assert metric in _scoring
            assert callable(_scoring[metric])
            assert float(_scoring[metric]([0,1,0,1],[1,0,0,1]))

    # END scoring v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^


    # n_jobs v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v
    # n_jobs: Optional[Union[int,None]]=None

    @pytest.mark.parametrize('junk_n_jobs',
        (-2, 0, 3.14, True, False, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_n_jobs(
        self, X_np, y_np, base_gstcv, junk_n_jobs
    ):

        base_gstcv.set_params(n_jobs=junk_n_jobs)

        with pytest.raises((TypeError, ValueError)):
            base_gstcv.fit(X_np, y_np)


    @pytest.mark.parametrize('bad_njobs', (-2, 0, 3.14))
    def test_n_jobs_rejects_bad_int(
        self, base_gstcv, bad_njobs, X_np, y_np
    ):

        with pytest.raises(ValueError):
            base_gstcv.set_params(n_jobs=bad_njobs).fit(X_np, y_np)


    def test_n_jobs_None_returns_None(self, base_gstcv, X_np, y_np):

        assert isinstance(
            base_gstcv.set_params(n_jobs=None).fit(X_np, y_np),
            type(base_gstcv)
        )

        assert base_gstcv.get_params(deep=True)['n_jobs'] is None


    @pytest.mark.parametrize('good_njobs', (-1, 2))
    def test_n_jobs_otherwise_returns_given(
        self, base_gstcv, good_njobs, X_np, y_np
    ):
        assert isinstance(
            base_gstcv.set_params(n_jobs=good_njobs).fit(X_np, y_np),
            type(base_gstcv)
        )

        assert base_gstcv.get_params(deep=True)['n_jobs'] == good_njobs

    # END n_jobs v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v


    # return_train_score v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v
    # return_train_score: Optional[bool]=False

    @pytest.mark.parametrize('junk_return_train_score',
        (-1, 0, 1, 3.14, None, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_return_train_score(
        self, X_np, y_np, base_gstcv, junk_return_train_score
    ):

        base_gstcv.set_params(return_train_score=junk_return_train_score)

        with pytest.raises(TypeError):
            base_gstcv.fit(X_np, y_np)


    @pytest.mark.parametrize('good_train_score', (True, False))
    def test_train_score_accepts_bool(
        self, base_gstcv, good_train_score, X_np, y_np
    ):

        base_gstcv.set_params(return_train_score=good_train_score)
        assert isinstance(base_gstcv.fit(X_np, y_np), type(base_gstcv))

        assert base_gstcv.get_params(deep=True)['return_train_score'] == \
               good_train_score

    # END return_train_score v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v


    # pre_dispatch v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v
    @pytest.mark.parametrize('junk_pre_dispatch',
        (-2, 0, False, 'trash', min, [0, 1], (0, 1), {0, 1},
         {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_pre_dispatch(
        self, X_np, y_np, base_gstcv, junk_pre_dispatch
    ):

        base_gstcv.set_params(pre_dispatch=junk_pre_dispatch)

        # this is raised by joblib, let it raise whatever
        with pytest.raises(Exception):
            base_gstcv.fit(X_np, y_np)

    # END pre_dispatch v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v






