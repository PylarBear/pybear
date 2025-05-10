# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.preprocessing import InterceptManager as IM

from pybear.utilities import check_pipeline

from uuid import uuid4

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression






class TestPipeline:


    @staticmethod
    @pytest.fixture()
    def _kwargs():
        return {
            'keep': 'first',
            'equal_nan': True,
            'rtol': 1e-5,
            'atol': 1e-8,
            'n_jobs': 1
        }


    @pytest.mark.parametrize('_format', ('np', 'pd'))
    def test_accuracy_in_pipe_vs_out_of_pipe(
        self, _X_factory, _shape, _kwargs, _format
    ):

        # this also incidentally tests functionality in a pipe

        # make a pipe of OneHotEncoder, IM, and LinearRegression
        # the X object needs to contain categorical data
        # fit the data on the pipeline, get coef_
        # fit the data on the steps severally, compare coef_


        _X = _X_factory(
            _dupl=None,
            _format=_format,
            _has_nan=False,
            _columns=[str(uuid4())[:5] for _ in range(_shape[1])],
            _dtype='obj',
            _shape=_shape
        )

        _y = np.random.uniform(0,1, _shape[0])

        # pipe ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        pipe = Pipeline(
            steps = [
                ('onehot', OneHotEncoder(sparse_output=True)),
                ('IM', IM(**_kwargs)),
                ('MLR', LinearRegression(fit_intercept = True, n_jobs = -1))
            ]
        )

        check_pipeline(pipe)

        pipe.fit(_X, _y)

        _coef_pipe = pipe.steps[2][1].coef_

        # END pipe ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


        # separate ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        encoded_X = OneHotEncoder(sparse_output=True).fit_transform(_X)
        deconstanted_X = IM(**_kwargs).fit_transform(encoded_X)
        mlr = LinearRegression(fit_intercept = True, n_jobs = -1)

        mlr.fit(deconstanted_X, _y)

        _coef_separate = mlr.coef_

        # END separate ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        assert np.allclose(_coef_pipe, _coef_separate)
















