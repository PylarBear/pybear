# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

from pybear.preprocessing import InterceptManager as IM
from pybear.utilities import check_pipeline



class TestPipeline:


    def test_accuracy_in_pipe_vs_out_of_pipe(
        self, _X_factory, _shape, _kwargs, y_np
):

        # this also incidentally tests functionality in a pipe

        # make a pipe of OneHotEncoder, IM, and LinearRegression
        # the X object needs to contain categorical data
        # fit the data on the pipeline, get coef_
        # fit the data on the steps severally, compare coef_

        _X = _X_factory(
            _dupl=None,
            _format='np',
            _has_nan=False,
            _columns=None,
            _dtype='obj',
            _shape=_shape
        )

        # pipe ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        pipe = Pipeline(
            steps = [
                ('onehot', OneHotEncoder(sparse_output=True)),
                ('IM', IM(**_kwargs)),
                ('MLR', LogisticRegression())
            ]
        )

        check_pipeline(pipe)

        pipe.fit(_X, y_np)

        _coef_pipe = pipe.steps[2][1].coef_

        # END pipe ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


        # separate ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        encoded_X = OneHotEncoder(sparse_output=True).fit_transform(_X)
        deconstanted_X = IM(**_kwargs).fit_transform(encoded_X)
        _coef_separate = LogisticRegression().fit(deconstanted_X, y_np).coef_

        # END separate ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        assert np.allclose(_coef_pipe, _coef_separate)





