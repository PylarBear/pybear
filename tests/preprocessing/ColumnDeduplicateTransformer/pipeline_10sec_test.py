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

from pybear.preprocessing import ColumnDeduplicateTransformer as CDT
from pybear.utilities import check_pipeline



class TestPipeline:


    def test_accuracy_in_pipe_vs_out_of_pipe(
        self, _X_factory, _shape, _kwargs, y_np
    ):

        # this also incidentally tests functionality in a pipe

        # make a pipe of OneHotEncoder, CDT, and LogisticRegression
        # the X object needs to contain categorical data
        # fit the data on the pipeline, get coef_
        # fit the data on the steps severally, compare coef_

        _X = _X_factory(
            _dupl=[[0, 2], [5, 7, 9]],
            _format='np',
            _has_nan=False,
            _columns=None,
            _dtype='obj',
            _shape=_shape
        )

        # pipe ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # n_jobs confliction doesnt seem to matter
        pipe = Pipeline(
            steps = [
                ('onehot', OneHotEncoder(sparse_output=True)),
                ('cdt', CDT(**_kwargs)),
                ('MLR', LogisticRegression())
            ]
        )

        check_pipeline(pipe)

        pipe.fit(_X, y_np)

        _coef_pipe = pipe.steps[2][1].coef_

        # END pipe ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


        # separate ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # n_jobs confliction doesnt seem to matter
        encoded_X = OneHotEncoder(sparse_output=True).fit_transform(_X)
        deduplicated_X = CDT(**_kwargs).fit_transform(encoded_X)
        _coef_separate = LogisticRegression().fit(deduplicated_X, y_np).coef_

        # END separate ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        assert np.allclose(_coef_pipe, _coef_separate)





