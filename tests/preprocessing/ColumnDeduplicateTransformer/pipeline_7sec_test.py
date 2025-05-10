# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.preprocessing import ColumnDeduplicateTransformer as CDT

from pybear.utilities import check_pipeline

from uuid import uuid4

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression






class TestPipeline:


    @staticmethod
    @pytest.fixture()
    def _dupls():
        return [
            [0, 2],
            [5, 7, 9]
        ]


    @staticmethod
    @pytest.fixture()
    def _kwargs():
        return {
            'keep': 'first',
            'conflict': 'ignore',
            'do_not_drop': None,
            'rtol': 1e-5,
            'atol': 1e-8,
            'equal_nan': True,
            'n_jobs': 1    # leave set at 1 because of confliction
        }


    @pytest.mark.parametrize('_format', ('np', 'pd'))
    def test_accuracy_in_pipe_vs_out_of_pipe(
        self, _X_factory, _dupls, _shape, _kwargs, _format
    ):

        # this also incidentally tests functionality in a pipe

        # make a pipe of OneHotEncoder, CDT, and LinearRegression
        # the X object needs to contain categorical data
        # fit the data on the pipeline, get coef_
        # fit the data on the steps severally, compare coef_


        _X = _X_factory(
            _dupl=_dupls,
            _format=_format,
            _has_nan=False,
            _columns=[str(uuid4())[:5] for _ in range(_shape[1])],
            _dtype='obj',
            _shape=_shape
        )

        _y = np.random.uniform(0,1, _shape[0])

        # pipe ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # n_jobs confliction doesnt seem to matter
        pipe = Pipeline(
            steps = [
                ('onehot', OneHotEncoder()),
                ('cdt', CDT(**_kwargs)),
                ('MLR', LinearRegression(fit_intercept = True, n_jobs = -1))
            ]
        )

        check_pipeline(pipe)

        pipe.fit(_X, _y)

        _coef_pipe = pipe.steps[2][1].coef_

        # END pipe ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


        # separate ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # n_jobs confliction doesnt seem to matter
        encoded_X = OneHotEncoder().fit_transform(_X)
        deduplicated_X = CDT(**_kwargs).fit_transform(encoded_X)
        mlr = LinearRegression(fit_intercept = True, n_jobs = -1)

        mlr.fit(deduplicated_X, _y)

        _coef_separate = mlr.coef_

        # END separate ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        assert np.allclose(_coef_pipe, _coef_separate)
















