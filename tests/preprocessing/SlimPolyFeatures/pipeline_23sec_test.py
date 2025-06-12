# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

from pybear.preprocessing import SlimPolyFeatures as SlimPoly
from pybear.utilities import check_pipeline



class TestPipeline:


    @pytest.mark.parametrize('ohe_sparse_output', (True, False))
    def test_accuracy_in_pipe_vs_out_of_pipe(
        self, _X_factory, _shape, _kwargs, y_np, ohe_sparse_output
    ):

        # this also incidentally tests functionality in a pipe

        # make a pipe of OneHotEncoder, SlimPoly, and LinearRegression
        # the X object needs to contain categorical data
        # fit the data on the pipeline, get coef_
        # fit the data on the steps severally, compare coef_

        _X = _X_factory(
            _dupl=None,
            _format='np',
            _has_nan=False,
            _columns=None,
            _dtype='obj',      # <+=== THIS MUST BE OBJ, NEED CATEGORICAL
            _shape=_shape
        )

        # pipe ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
        pipe = Pipeline(
            steps = [
                ('onehot', OneHotEncoder(sparse_output=ohe_sparse_output)),
                ('SlimPoly', SlimPoly(**_kwargs)),
                ('MLR', LinearRegression(fit_intercept = True, n_jobs = 1))
            ]
        )

        check_pipeline(pipe)

        pipe.fit(_X, y_np)

        _coef_pipe = pipe.steps[2][1].coef_

        # END pipe ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


        # separate ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

        encoded_X = OneHotEncoder(sparse_output=False).fit_transform(_X)
        deduplicated_X = SlimPoly(**_kwargs).fit_transform(encoded_X)
        _coef_separate = LinearRegression(
            fit_intercept = True, n_jobs = 1
        ).fit(deduplicated_X, y_np).coef_

        # END separate ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        assert np.allclose(_coef_pipe, _coef_separate)





