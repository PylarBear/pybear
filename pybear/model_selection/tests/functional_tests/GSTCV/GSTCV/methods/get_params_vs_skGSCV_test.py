# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest

import numpy as np

from sklearn.model_selection import GridSearchCV as sk_GridSearchCV
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression
from sklearn.preprocessing import OneHotEncoder as sk_OneHotEncoder
from sklearn.pipeline import Pipeline

from model_selection.GSTCV._GSTCV.GSTCV import GSTCV



class TestSKGetParams:

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

        gstcv = GSTCV(
            sk_LogisticRegression(**logistic_params),
            logistic_param_grid
        )

        with pytest.raises(ValueError):
            gstcv.get_params(bad_deep)


    def test_one_est(self, logistic_params, logistic_param_grid):

        # set up GS(T)CV's ** * ** * ** * ** * ** * ** * ** * ** * ** *
        estimator = sk_LogisticRegression(**logistic_params)
        param_grid = logistic_param_grid

        skgscv = sk_GridSearchCV(estimator=estimator, param_grid=param_grid)

        gstcv = GSTCV(estimator=estimator, param_grid=param_grid)
        # END set up GS(T)CV's ** * ** * ** * ** * ** * ** * ** * ** *

        # test shallow ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # only test params' names, not values; GSTCV's defaults may be
        # different than skGSCV's

        skgscv_shallow = list(skgscv.get_params(deep=False).keys())
        gstcv_shallow = list(gstcv.get_params(deep=False).keys())

        assert 'thresholds' in gstcv_shallow

        # +1 for thresholds
        assert len(gstcv_shallow) == len(skgscv_shallow) + 1

        gstcv_shallow.remove('thresholds')

        assert np.array_equiv(skgscv_shallow, gstcv_shallow)

        # END test shallow ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # test deep ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # only test params' names, not values; GSTCV's defaults my be
        # different than skGSCV's

        skgscv_deep = list(skgscv.get_params(deep=True).keys())
        gstcv_deep = list(gstcv.get_params(deep=True).keys())

        assert 'thresholds' in gstcv_deep

        # +1 for thresholds
        assert len(gstcv_deep) == len(skgscv_deep) + 1

        gstcv_deep.remove('thresholds')

        assert np.array_equiv(skgscv_deep, gstcv_deep)

        # END test deep ** * ** * ** * ** * ** * ** * ** * ** * ** *


    def test_pipe_shallow(self, logistic_params, pipe_param_grid):

        # set up GS(T)CV's ** * ** * ** * ** * ** * ** * ** * ** * ** *
        pipe = Pipeline(
            steps=[
                ('onehot', sk_OneHotEncoder(sparse_output=False)),
                ('logistic', sk_LogisticRegression(**logistic_params))
            ]
        )

        skgscv = sk_GridSearchCV(estimator=pipe, param_grid=pipe_param_grid)
        assert skgscv.estimator.steps[0][0] == 'onehot'
        assert skgscv.estimator.steps[1][0] == 'logistic'

        gstcv = GSTCV(estimator=pipe, param_grid=pipe_param_grid)
        assert gstcv.estimator.steps[0][0] == 'onehot'
        assert gstcv.estimator.steps[1][0] == 'logistic'
        # END set up GS(T)CV's ** * ** * ** * ** * ** * ** * ** * ** *

        # test shallow ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # only test params' names, not values; GSTCV's defaults my be
        # different than skGSCV's

        skgscv_shallow = list(skgscv.get_params(deep=False).keys())
        gstcv_shallow = list(gstcv.get_params(deep=False).keys())

        assert 'thresholds' in gstcv_shallow

        # +1 for thresholds
        assert len(gstcv_shallow) == len(skgscv_shallow) + 1

        assert len(skgscv_shallow) == 10
        assert len(gstcv_shallow) == 11

        gstcv_shallow.remove('thresholds')

        assert np.array_equiv(skgscv_shallow, gstcv_shallow)

        # END test shallow ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # test deep ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    def test_pipe_deep(self, logistic_params, pipe_param_grid):

        pipe = Pipeline(
            steps=[
                ('onehot', sk_OneHotEncoder(sparse_output=False)),
                ('logistic', sk_LogisticRegression(**logistic_params))
            ]
        )

        # set up GS(T)CV's ** * ** * ** * ** * ** * ** * ** * ** * ** *
        skgscv = sk_GridSearchCV(estimator=pipe, param_grid=pipe_param_grid)
        assert skgscv.estimator.steps[0][0] == 'onehot'
        assert skgscv.estimator.steps[1][0] == 'logistic'

        gstcv = GSTCV(estimator=pipe, param_grid=pipe_param_grid)
        assert gstcv.estimator.steps[0][0] == 'onehot'
        assert gstcv.estimator.steps[1][0] == 'logistic'
        # END set up GS(T)CV's ** * ** * ** * ** * ** * ** * ** * ** *

        # test deep ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # only test params' names, not values; GSTCV's defaults my be
        # different than skGSCV's

        skgscv_deep = list(skgscv.get_params(deep=True).keys())
        gstcv_deep = list(gstcv.get_params(deep=True).keys())

        assert 'thresholds' in gstcv_deep

        # +1 for thresholds
        assert len(gstcv_deep) == len(skgscv_deep) + 1

        assert len(skgscv_deep) == 38
        assert len(gstcv_deep) == 39

        gstcv_deep.remove('thresholds')

        assert np.array_equiv(skgscv_deep, gstcv_deep)

        # END test deep ** * ** * ** * ** * ** * ** * ** * ** * ** *










