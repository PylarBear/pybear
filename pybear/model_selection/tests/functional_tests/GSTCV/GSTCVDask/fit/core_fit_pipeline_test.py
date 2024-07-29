# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np
import dask.array as da
import pandas as pd
import distributed
from dask_ml.preprocessing import OneHotEncoder as dask_OneHotEncoder
from xgboost.dask import XGBClassifier as dask_XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from dask_ml.model_selection import (
    GridSearchCV as dask_GridSearchCV,
    KFold as dask_KFold
)

from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score
)

from model_selection.GSTCV._fit_shared._cv_results._cv_results_builder \
    import _cv_results_builder

from model_selection.GSTCV._GSTCVDask._fit._core_fit import _core_fit



# 24_07_14 this module tests the equality of GSTCVDask's cv_results_ with
# 0.5 threshold against dask_ml GSCV cv_results_ when using sk Pipeline
@pytest.skip(reason=f'test is not finishing, needs speedup', allow_module_level=True)
class TestCoreFitPipelineAccuracy:

    # def _core_fit(
    #         _X: XDaskWIPType,
    #         _y: YDaskWIPType,
    #         _estimator: ClassifierProtocol,
    #         _cv_results: CVResultsType,
    #         _cv: Union[int, GenericKFoldType],
    #         _error_score: Union[int, float, Literal['raise']],
    #         _verbose: int,
    #         _scorer: ScorerWIPType,
    #         _cache_cv: bool,
    #         _iid: bool,
    #         _return_train_score: bool,
    #         _PARAM_GRID_KEY: npt.NDArray[np.uint8],
    #         _THRESHOLD_DICT: dict[int, npt.NDArray[np.float64]],
    #         **params
    # ) -> CVResultsType


    # fixtures ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    @staticmethod
    @pytest.fixture
    def good_X():
        return da.random.choice(list('abcdefghijklmnop'), (1000, 5), replace=True)


    @staticmethod
    @pytest.fixture
    def good_y():
        return da.random.randint(0, 2, (1000,))


    @staticmethod
    @pytest.fixture
    def good_estimator():
        return Pipeline(
            steps=[
                ('OneHot', dask_OneHotEncoder()),
                ('XGBClassifier', dask_XGBClassifier())
            ]
        )












    @staticmethod
    @pytest.fixture
    def good_param_grid():
        return [
            {'OneHot__min_frequency': [5,10], 'DaskLogistic__C': np.logspace(-3,3,7)},
            {'OneHot__min_frequency': [25,30], 'DaskLogistic__C': np.logspace(-5,-1,6)}
        ]


    @staticmethod
    @pytest.fixture
    def helper_for_cv_results_and_PARAM_GRID_KEY(good_cv_int, good_scorer,
        good_param_grid):

        out_cv_results, out_key = _cv_results_builder(
            # DO NOT PUT 'thresholds' IN PARAM GRIDS!
            param_grid=good_param_grid,
            cv=good_cv_int,
            scorer=good_scorer,
            return_train_score=True
        )

        param_grid = ParameterGrid(good_param_grid)
        out_cv_results['params'] = np.ma.masked_array(param_grid)
        del param_grid

        return out_cv_results, out_key


    @staticmethod
    @pytest.fixture
    def good_cv_results(helper_for_cv_results_and_PARAM_GRID_KEY):

        return helper_for_cv_results_and_PARAM_GRID_KEY[0]


    @staticmethod
    @pytest.fixture
    def good_cv_int():
        return 5


    @staticmethod
    @pytest.fixture
    def good_cv_iter(good_cv_int, good_X, good_y):
        return dask_KFold(n_splits=good_cv_int).split(good_X, good_y)


    @staticmethod
    @pytest.fixture
    def good_error_score():
        return 'raise'


    @staticmethod
    @pytest.fixture
    def good_scorer():
        return {
            'precision': precision_score,
            'recall': recall_score,
            'accuracy': accuracy_score,
            'balanced_accuracy': balanced_accuracy_score
        }


    @staticmethod
    @pytest.fixture
    def good_cache_cv():
        return True


    @staticmethod
    @pytest.fixture
    def good_iid():
        return True


    @staticmethod
    @pytest.fixture
    def good_PARAM_GRID_KEY(helper_for_cv_results_and_PARAM_GRID_KEY):

        return helper_for_cv_results_and_PARAM_GRID_KEY[1]


    @staticmethod
    @pytest.fixture
    def good_THRESHOLD_DICT():
        return {0: np.linspace(0,1,21), 1: np.linspace(0,1,11)}


    @staticmethod
    @pytest.fixture(scope='module')
    def _client():
        client = distributed.Client(n_workers=1, threads_per_worker=1)
        yield client
        client.close()

    # END fixtures ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **



    @pytest.mark.parametrize('_n_jobs', (-1, 1))  # 1 is important
    def test_accuracy_cv_int_vs_cv_iter(self, good_X, good_y, good_estimator,
        good_cv_results, good_cv_int, good_cv_iter, good_error_score,
        good_scorer, good_cache_cv, good_iid, good_PARAM_GRID_KEY,
        good_THRESHOLD_DICT, _n_jobs, _client):

        # test equivalent cv as int or iterable give same output

        with _client:
            out_int = _core_fit(
                good_X,
                good_y,
                good_estimator,
                good_cv_results,
                good_cv_int,   # <===============
                good_error_score,
                0, # good_verbose,
                good_scorer,
                good_cache_cv,
                good_iid,
                True, # good_return_train_score,
                good_PARAM_GRID_KEY,
                good_THRESHOLD_DICT
            )

        with _client:
            out_iter = _core_fit(
                good_X,
                good_y,
                good_estimator,
                good_cv_results,
                good_cv_iter,   # <===============
                good_error_score,
                0, # good_verbose,
                good_scorer,
                good_cache_cv,
                good_iid,
                True, # good_return_train_score,
                good_PARAM_GRID_KEY,
                good_THRESHOLD_DICT
            )

        assert pd.DataFrame(data=out_int).equals(pd.DataFrame(data=out_iter))



    # this is just an example, the columns are not correct for these test!
    # 'mean_fit_time'
    # 'std_fit_time'
    # 'mean_score_time'
    # 'std_score_time'
    # 'param_C'
    # 'param_solver'
    # 'params'
    # 'best_threshold_accuracy'
    # 'split0_test_accuracy'
    # 'split1_test_accuracy'
    # 'mean_test_accuracy'
    # 'std_test_accuracy'
    # 'rank_test_accuracy'
    # 'split0_train_accuracy'
    # 'split1_train_accuracy'
    # 'mean_train_accuracy'
    # 'std_train_accuracy'
    # 'best_threshold_balanced_accuracy'
    # 'split0_test_balanced_accuracy'
    # 'split1_test_balanced_accuracy'
    # 'mean_test_balanced_accuracy'
    # 'std_test_balanced_accuracy'
    # 'rank_test_balanced_accuracy'
    # 'split0_train_balanced_accuracy'
    # 'split1_train_balanced_accuracy'
    # 'mean_train_balanced_accuracy'
    # 'std_train_balanced_accuracy'




    @pytest.mark.parametrize('_scorer',
        (
            {
                 'precision': precision_score,
                 'recall': recall_score,
                 'accuracy': accuracy_score,
                 'balanced_accuracy': balanced_accuracy_score
            },
            {
                'accuracy': accuracy_score,
                'balanced_accuracy': balanced_accuracy_score
            }
        )
    )
    @pytest.mark.parametrize('_param_grid',
        (
            [
                {'OneHot__min_frequency': [5, 10], 'DaskLogistic__C': [.001, .01]}
            ],
            [
                {'OneHot__min_frequency': [5, 10], 'DaskLogistic__C': [.001, .01]},
                {'OneHot__min_frequency': [25, 30], 'DaskLogistic__C': [.0001, .001]}
            ],
            [
                {'DaskLogistic__C': [.0001, .001]},
                {'OneHot__min_frequency': [25, 30], 'DaskLogistic__C': [.001, .01]}
            ],
        )
    )
    @pytest.mark.parametrize('_return_train_score', (True, False))
    @pytest.mark.parametrize('_cache_cv', (True, False))
    @pytest.mark.parametrize('_iid', (True, False))
    def test_accuracy_vs_dask_gscv(self, good_X, good_y, good_estimator,
        good_cv_int, good_error_score, _scorer,  _cache_cv, _iid,
        _return_train_score, _param_grid, _client):


        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        good_cv_results, PARAM_GRID_KEY = _cv_results_builder(
            param_grid=_param_grid,
            cv=good_cv_int,
            scorer=_scorer,
            return_train_score=_return_train_score
        )

        with _client:
            gstcv_cv_results = _core_fit(
                good_X,
                good_y,
                good_estimator,
                good_cv_results,
                good_cv_int,
                good_error_score,
                0, # good_verbose,
                _scorer,
                _cache_cv,
                _iid,
                _return_train_score,
                PARAM_GRID_KEY,
                {i: np.array([0.5]) for i in range(len(_param_grid))} # THRESHOLD_DICT
            )

        pd_gstcv_cv_results = pd.DataFrame(gstcv_cv_results)
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        __scorer = {k: make_scorer(v) for k,v in _scorer.items()}

        out_dask_gscv = dask_GridSearchCV(
            good_estimator,
            _param_grid,
            cv=good_cv_int,
            error_score=good_error_score,
            verbose=0,
            scoring=__scorer,
            n_jobs=None,
            cache_cv=_cache_cv,
            iid=_iid,
            return_train_score=_return_train_score,
            refit=list(_scorer.keys())[0]
        )

        with _client:
            out_dask_gscv.fit(good_X, good_y)

        dask_cv_results = out_dask_gscv.cv_results_

        pd_dask_cv_results = pd.DataFrame(dask_cv_results)
        # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **



        assert len(pd_gstcv_cv_results) == len(pd_dask_cv_results), \
            f"different rows in cv_results_"

        MASK = list(map(lambda x: 'threshold' in x, pd_gstcv_cv_results.columns))
        __ = pd_gstcv_cv_results.drop(columns=pd_gstcv_cv_results.columns[MASK])
        assert np.array_equiv(__.columns, pd_dask_cv_results.columns), \
            f'columns not equal / out of order'
        del MASK, __

        for column in pd_gstcv_cv_results:

            if 'threshold' not in column:
                assert column in pd_dask_cv_results, \
                    print(f'\033[91mcolumn {column} not in!\033[0m')
            elif 'threshold' in column:
                assert (pd_gstcv_cv_results[column] == 0.5).all()
                continue

            if 'time' in column:
                assert (pd_gstcv_cv_results[column] > 0).all()
                assert (gstcv_cv_results[column] > 0).all()
                continue


            _are_floats = False

            MASK = np.logical_not(pd_gstcv_cv_results[column].isna())

            try:
                _gstcv_out = pd_gstcv_cv_results[column][MASK].to_numpy(
                    dtype=np.float64
                )
                _dask_out = pd_dask_cv_results[column][MASK].to_numpy(
                    dtype=np.float64
                )

                _are_floats = True

            except:
                # check param columns
                _gstcv_out = pd_gstcv_cv_results[column][MASK].to_numpy(),
                _dask_out = pd_dask_cv_results[column][MASK].to_numpy()


            if _are_floats:
                assert (_gstcv_out > 0).any()
                assert (_dask_out > 0).any()

                assert np.allclose(_gstcv_out, _dask_out, atol=0.00001)

            elif not _are_floats:
                # check param columns
                assert np.array_equiv(_gstcv_out, _dask_out)





























