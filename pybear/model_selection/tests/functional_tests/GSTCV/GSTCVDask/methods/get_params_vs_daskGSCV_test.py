# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest

import numpy as np

from dask_ml.model_selection import GridSearchCV as dask_GridSearchCV
from dask_ml.linear_model import LogisticRegression as dask_LogisticRegression
from dask_ml.preprocessing import OneHotEncoder as dask_OneHotEncoder
from sklearn.pipeline import Pipeline

from model_selection.GSTCV._GSTCVDask.GSTCVDask import GSTCVDask



class TestDaskGetParams:

    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @staticmethod
    @pytest.fixture(scope='class')
    def logistic_param_grid():
        return {
            'C': np.logspace(-5,-1,5),
            'tol': [1e-6, 1e-5, 1e-4]
        }


    @staticmethod
    @pytest.fixture(scope='class')
    def logistic_params():
        return {'C': 13, 'tol': 1e-6, 'solver': 'lbfgs'}


    @staticmethod
    @pytest.fixture(scope='class')
    def pipe_param_grid():
        return {
            'onehot__min_frequency': [4,5,6],
            'onehot__max_categories': [2,3,4],
            'logistic__C': np.logspace(-5,-1,5),
            'logistic__tol': [1e-6, 1e-5, 1e-4]
        }

    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    @pytest.mark.parametrize('bad_deep',
        (0, 1, 3.14, None, 'trash', min, [0,1], (0,1), {'a':1}, lambda x: x)
    )
    def test_rejects_non_bool_deep(self, bad_deep, logistic_params,
                                   logistic_param_grid):

        gstcv = GSTCVDask(
            dask_LogisticRegression(**logistic_params),
            logistic_param_grid
        )

        with pytest.raises(ValueError):
            gstcv.get_params(bad_deep)


    def test_one_est(self, logistic_params, logistic_param_grid):

        # set up GS(T)CV's ** * ** * ** * ** * ** * ** * ** * ** * ** *
        estimator = dask_LogisticRegression(**logistic_params)
        param_grid = logistic_param_grid

        daskgscv = dask_GridSearchCV(estimator=estimator, param_grid=param_grid)

        gstcv = GSTCVDask(estimator=estimator, param_grid=param_grid)
        # END set up GS(T)CV's ** * ** * ** * ** * ** * ** * ** * ** *

        # test shallow ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # only test params' names, not values; GSTCVDask's defaults my be
        # different than daskGSCV's

        daskgscv_shallow = list(daskgscv.get_params(deep=False).keys())
        gstcv_shallow = list(gstcv.get_params(deep=False).keys())

        assert 'thresholds' in gstcv_shallow
        assert 'verbose' in gstcv_shallow
        # +2 for thresholds / verbose
        assert len(gstcv_shallow) == len(daskgscv_shallow) + 2

        gstcv_shallow.remove('thresholds')
        gstcv_shallow.remove('verbose')
        assert np.array_equiv(daskgscv_shallow, gstcv_shallow)

        # END test shallow ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # test deep ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # only test params' names, not values; GSTCVDask's defaults my be
        # different than daskGSCV's

        daskgscv_deep = list(daskgscv.get_params(deep=True).keys())
        gstcv_deep = list(gstcv.get_params(deep=True).keys())

        assert 'thresholds' in gstcv_deep
        assert 'verbose' in gstcv_deep
        # +2 for thresholds / verbose
        assert len(gstcv_deep) == len(daskgscv_deep) + 2

        gstcv_deep.remove('thresholds')
        gstcv_deep.remove('verbose')
        assert np.array_equiv(daskgscv_deep, gstcv_deep)

        # END test deep ** * ** * ** * ** * ** * ** * ** * ** * ** *


    def test_pipe_shallow(self, logistic_params, pipe_param_grid):

        # set up GS(T)CV's ** * ** * ** * ** * ** * ** * ** * ** * ** *
        estimator = Pipeline(
            steps=[
                ('onehot', dask_OneHotEncoder(sparse_output=False)),
                ('logistic', dask_LogisticRegression(**logistic_params))
            ]
        )

        daskgscv = dask_GridSearchCV(estimator=estimator, param_grid=pipe_param_grid)
        assert daskgscv.estimator.steps[0][0] == 'onehot'
        assert daskgscv.estimator.steps[1][0] == 'logistic'

        gstcv = GSTCVDask(estimator=estimator, param_grid=pipe_param_grid)
        assert gstcv.estimator.steps[0][0] == 'onehot'
        assert gstcv.estimator.steps[1][0] == 'logistic'
        # END set up GS(T)CV's ** * ** * ** * ** * ** * ** * ** * ** *

        # test shallow ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # only test params' names, not values; GSTCVDask's defaults my be
        # different than daskGSCV's

        daskgscv_shallow = list(daskgscv.get_params(deep=False).keys())
        gstcv_shallow = list(gstcv.get_params(deep=False).keys())

        assert 'thresholds' in gstcv_shallow
        assert 'verbose' in gstcv_shallow
        # +1 for thresholds / verbose
        assert len(gstcv_shallow) == len(daskgscv_shallow) + 2

        assert len(daskgscv_shallow) ==11
        assert len(gstcv_shallow) == 13

        gstcv_shallow.remove('thresholds')
        gstcv_shallow.remove('verbose')
        assert np.array_equiv(daskgscv_shallow, gstcv_shallow)

        # END test shallow ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # test deep ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    def test_pipe_deep(self, logistic_params, pipe_param_grid):

        pipe = Pipeline(
            steps=[
                ('onehot', dask_OneHotEncoder(sparse_output=False)),
                ('logistic', dask_LogisticRegression(**logistic_params))
            ]
        )

        # set up GS(T)CV's ** * ** * ** * ** * ** * ** * ** * ** * ** *
        daskgscv = dask_GridSearchCV(estimator=pipe, param_grid=pipe_param_grid)
        assert daskgscv.estimator.steps[0][0] == 'onehot'
        assert daskgscv.estimator.steps[1][0] == 'logistic'

        gstcv = GSTCVDask(estimator=pipe, param_grid=pipe_param_grid)
        assert gstcv.estimator.steps[0][0] == 'onehot'
        assert gstcv.estimator.steps[1][0] == 'logistic'
        # END set up GS(T)CV's ** * ** * ** * ** * ** * ** * ** * ** *

        # test deep ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # only test params' names, not values; GSTCVDask's defaults my be
        # different than daskGSCV's

        daskgscv_deep = list(daskgscv.get_params(deep=True).keys())
        gstcv_deep = list(gstcv.get_params(deep=True).keys())

        assert 'thresholds' in gstcv_deep
        assert 'verbose' in gstcv_deep
        # +2 for thresholds / verbose
        assert len(gstcv_deep) == len(daskgscv_deep) + 2

        assert len(daskgscv_deep) == 36
        assert len(gstcv_deep) == 38

        gstcv_deep.remove('thresholds')
        gstcv_deep.remove('verbose')
        assert np.array_equiv(daskgscv_deep, gstcv_deep)

        # END test deep ** * ** * ** * ** * ** * ** * ** * ** * ** *










