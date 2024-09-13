# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest
import numpy as np

from sklearn.model_selection import GridSearchCV as sklearn_GridSearchCV
from sklearn.linear_model import Ridge

from dask_ml.model_selection import GridSearchCV as dask_GridSearchCV
from dask_ml.linear_model import LogisticRegression

from pybear.model_selection.autogridsearch.autogridsearch_wrapper import \
    autogridsearch_wrapper




sklearn_estimator = Ridge(
    # alpha=1.0,  use this in AGSCV
    # fit_intercept=True, use this in AGSCV
    copy_X=True,
    # max_iter=None, use this in AGSCV
    tol=0.0001,
    # solver='auto',  use this in AGSCV
    positive=False,
    random_state=None
)



dask_estimator = LogisticRegression(
    penalty='l2',
    dual=False,
    tol=0.0001,
    # C=1.0,   use this in AGSCV
    # fit_intercept=True,   use this in AGSCVs
    intercept_scaling=1.0,
    class_weight=None,
    random_state=None,
    # solver='admm',  use this in AGSCV [lbfgs, admm]
    # max_iter=100,   use this in AGSCV
    multi_class='ovr',
    verbose=0,
    warm_start=False,
    n_jobs=1,
    solver_kwargs=None
)



class TestDemo:

    # prove out it does correct passes wrt total_passes/shifts/tpih
    # tests RESULTS_ & GRIDS_
    # shift_ctr

    @pytest.mark.parametrize('_package', ('sklearn', 'dask'))
    @pytest.mark.parametrize('_space, _gap',
        (
         ('linspace', 'na'),
         ('logspace', 1.0),
         ('logspace', 2.0),
        )
    )
    @pytest.mark.parametrize('_type', ('fixed', 'soft', 'hard'))
    @pytest.mark.parametrize('_univ_min_bound', (True, False))
    @pytest.mark.parametrize('_points', (3, 4))
    @pytest.mark.parametrize('_total_passes', (2, 5))
    @pytest.mark.parametrize('_shrink_pass', (2, 3, 1_000_000))
    @pytest.mark.parametrize('_max_shifts', (1, 3))
    @pytest.mark.parametrize(f'_tpih', (True, False))
    @pytest.mark.parametrize('_pass_best', (True, False))
    def test_sklearn(self, _package, _space, _gap, _type, _univ_min_bound,
         _points, _total_passes, _shrink_pass, _max_shifts, _tpih, _pass_best):

        if _package == 'sklearn':
            SKLearnAutoGridSearch = autogridsearch_wrapper(sklearn_GridSearchCV)
        elif _package == 'dask':
            DaskAutoGridSearch = autogridsearch_wrapper(dask_GridSearchCV)

        _POINTS = [_points for _ in range(_total_passes)]
        _POINTS[_shrink_pass-1:] = [1 for _ in _POINTS[_shrink_pass-1:]]

        _params = {
            'alpha': [[], _POINTS, _type + '_float'],
            'fit_intercept': [[True, False], _shrink_pass, 'bool'],
            'max_iter': [[], _POINTS, _type + '_integer'],
            'solver': [['lbfgs', 'saga'], _shrink_pass, 'string']
        }

        del _POINTS

        # build first grids ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # make lin univ min bound and log gap 1, then adjust as needed
        _alpha_lin_range = np.linspace(0, 10_000 * (_points - 1), _points)
        _alpha_log_range = np.linspace(0, _points - 1, _points)
        _max_iter_lin_range = np.linspace(1, 1_001 * (_points - 1), _points)
        _max_iter_log_range = np.linspace(0, _points - 1, _points)

        if _univ_min_bound:
            pass
        elif not _univ_min_bound:
            _alpha_lin_range += 10_000
            _max_iter_lin_range += 999
            _max_iter_log_range += 1

        if _gap == 2:
            _alpha_log_range *= 2
            _max_iter_log_range *= 2

        if _space == 'linspace':
            _alpha_grid = _alpha_lin_range
            _max_iter_grid = _max_iter_lin_range
        elif _space == 'logspace':
            _alpha_grid = np.power(10, _alpha_log_range)
            _max_iter_grid = np.power(10, _alpha_log_range)


        try:
            _params['alpha'][0] = list(map(float, _alpha_grid))
        except:
            pass

        try:
            _params['max_iter'][0] = list(map(int, _max_iter_grid))
        except:
            pass
        # END build first grids ** * ** * ** * ** * ** * ** * ** * ** *

        if _package == 'sklearn':
            test_cls = SKLearnAutoGridSearch(
                sklearn_estimator,
                params=_params,
                total_passes=_total_passes,
                total_passes_is_hard=_tpih,
                max_shifts=_max_shifts
            )
        elif _package == 'dask':
            # 24_05_32 params was originally built for sklearn Ridge, but
            # in trying to expand tests to dask, dask doesnt have Ridge and
            # dask logistic takes 'C' instead of 'alpha'. Swap 'C' into
            # params in place of 'alpha' for dask tests.
            _params['C'] = _params['alpha']
            del _params['alpha']

            # also swap the engines in solver, the only engine that matches
            # between sklearn and dask is 'lbfgs'
            _params['solver'][0] = ['lbfgs', 'admm']

            test_cls = DaskAutoGridSearch(
                dask_estimator,
                params=_params,
                total_passes=_total_passes,
                total_passes_is_hard=_tpih,
                max_shifts=_max_shifts
            )


        # build _true_best_params ** * ** * ** * ** * ** * ** * ** * **
        # arbitrary values in _true_best_params ** * ** * ** * ** * ** *
        __ = {}
        try:
            _params['alpha']    # for sklearn ridge, may have been swapped out
            if _type == 'soft':
                __['alpha'] = 53_827
            elif _type == 'hard':
                x =  _params['alpha'][0]
                __['alpha'] = float(np.mean((x[-2], x[-1])))
                del x
            elif _type == 'fixed':
                __['alpha'] = _params['alpha'][0][-2]

        except:
            pass

        try:
            _params['C']   # for dask logistic, may have been swapped in
            if _type == 'soft':
                __['C'] = 53_827
            elif _type == 'hard':
                x =  _params['C'][0]
                __['C'] = float(np.mean((x[-2], x[-1])))
                del x
            elif _type == 'fixed':
                __['C'] = _params['C'][0][-2]

        except:
            pass

        try:
            _params['fit_intercept']
            __['fit_intercept'] = True
        except:
            pass

        try:
            _params['max_iter']
            if _type == 'soft':
                __['max_iter'] = 8_607
            elif _type == 'hard':
                x =  _params['max_iter'][0]
                __['max_iter'] = float(np.floor(np.mean((x[-2], x[-1]))))
                del x
            elif _type == 'fixed':
                __['max_iter'] = _params['max_iter'][0][-2]
        except:
            pass

        try:
            _params['solver']
            __['solver'] = np.random.choice(_params['solver'][0], 1)[0]
        except:
            pass

        _true_best_params = __
        # END arbitrary values in _true_best_params ** * ** * ** * ** *

        _test_cls = test_cls.demo(
            true_best_params=_true_best_params if _pass_best else None,
            mock_gscv_pause_time=0
        )

        del test_cls

        # assertions ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

        # 'params'
        assert _test_cls.params.keys() == _params.keys()
        for _param in _params:
            assert _test_cls.params[_param][0] == _params[_param][0]
            if _params[_param][-1] in ['string', 'bool']:
                assert _test_cls.params[_param][2] == _params[_param][2]
            else:
                assert len(_test_cls.params[_param][0]) == \
                                    _test_cls.params[_param][1][0]
                assert len(_test_cls.params[_param][1]) == _test_cls.total_passes

        del _param

        # 'total_passes'
        assert _test_cls.total_passes >= _total_passes

        # 'total_passes_is_hard'
        assert _test_cls.total_passes_is_hard == _tpih

        # 'max_shifts'
        assert _test_cls.max_shifts == _max_shifts

        # 'GRIDS_'
        assert list(_test_cls.GRIDS_.keys()) == list(range(_test_cls.total_passes))
        for _pass_ in _test_cls.GRIDS_:
            assert _test_cls.GRIDS_[_pass_].keys() == _params.keys()
            assert all(map(isinstance,
                           _test_cls.GRIDS_[_pass_].values(),
                           (list for _ in _test_cls.GRIDS_[_pass_]))
                       )
            for _param_ in _params:
                __ = _test_cls.GRIDS_[_pass_][_param_]
                if _params[_param_][-1] in ['string', 'bool']:
                    # 'shrink pass' may have been incremented by shifts,
                    # which would show in _test_cls.params, but not the
                    # _params in this scope
                    if _pass_ >= _test_cls.params[_param_][-2] - 1:
                        assert len(__) == 1
                else:
                    assert len(__) == _test_cls.params[_param_][1][_pass_]
            del _param_, __
        del _pass_

        # 'RESULTS_'
        assert list(_test_cls.RESULTS_.keys()) == list(range(_test_cls.total_passes))
        for _pass_ in _test_cls.RESULTS_:
            assert _test_cls.RESULTS_[_pass_].keys() == _params.keys()
            assert all(
                map(
                    isinstance,
                    _test_cls.RESULTS_[_pass_].values(),
                    ((int, float, bool, str) for _ in _test_cls.RESULTS_[_pass_])
                )
            )
        del _pass_

        _last_param_grid = _test_cls.GRIDS_[max(_test_cls.GRIDS_.keys())]
        _last_best = _test_cls.RESULTS_[max(_test_cls.RESULTS_.keys())]
        for _param in _params:
            _last_grid = _last_param_grid[_param]
            if any([_ in _params[_param][-1] for _ in ['string', 'fixed']]):
                assert _last_best[_param] in _params[_param][0]
                assert _last_best[_param] in _last_grid
                # remember demo has 10% chance that these could be non-best
                # if _pass_best:
                #     assert _true_best_params[_param] in _last_grid
                #     assert _last_best[_param] == _true_best_params[_param]
            else:
                # when shifting these, both may not be true
                 assert _true_best_params[_param] >= min(_last_grid) or \
                            _true_best_params[_param] <= max(_last_grid)

        del _last_param_grid, _last_best, _param, _last_grid


    @pytest.mark.skip(reason=f"not done yet")
    def test_dask_w_true_best_params(self):

        DaskAutoGridSearch = autogridsearch_wrapper(dask_GridSearchCV)

        test_cls = DaskAutoGridSearch(
            estimator=dask_estimator,
            params=params,
            total_passes=2,
            total_passes_is_hard=True,
            max_shifts=2,
        )

        _true_best_params = {

        }

        out = test_cls.demo(
            true_best_params=_true_best_params,
            mock_gscv_pause_time=0
        )























