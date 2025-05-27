# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from uuid import uuid4

import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as ss

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    FunctionTransformer
)
from sklearn.linear_model import LinearRegression

from pybear.preprocessing._SlimPolyFeatures.SlimPolyFeatures import \
    SlimPolyFeatures as SlimPoly

from pybear.utilities import check_pipeline



class TestPipeline:


    @pytest.mark.parametrize('_format', ('np', 'pd', 'pl', 'csr', 'csc', 'bsr'))
    def test_accuracy_in_pipe_vs_out_of_pipe(
        self, _X_factory, _shape, _kwargs, _format
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
            _columns=[str(uuid4())[:5] for _ in range(_shape[1])],
            _dtype='obj',      # <+=== THIS MUST BE OBJ, NEED CATEGORICAL
            _shape=_shape
        )

        _y = np.random.uniform(0,1, _shape[0])


        # pipe ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # dont use OHE sparse_output, that only puts out CSR.
        # use FunctionTransformer to make & test different ss formats.

        def _convert_format_before_SPF(_X_):

            nonlocal _format
            # pizza what is _format for?
            # _format: Literal['np', 'pd', 'pl', 'csr', 'csc', 'bsr']

            # a function for FunctionTransformer, to convert X format
            # inside a pipeline, if needed.

            assert isinstance(_X_, np.ndarray)

            if _format == 'np':
                _X = _X_
            elif _format == 'pd':
                _X = pd.DataFrame(data=_X_)
            elif _format == 'pl':
                _X = pl.from_numpy(data=_X_)
            elif _format == 'csr':
                _X = ss.csr_array(_X_)
            elif _format == 'csc':
                _X = ss.csc_array(_X_)
            elif _format == 'bsr':
                _X = ss.bsr_array(_X_)
            else:
                raise Exception

            return _X


        pipe = Pipeline(
            steps = [
                ('onehot', OneHotEncoder(sparse_output=False)),
                (
                    'FunctionTransformer2',
                    FunctionTransformer(_convert_format_before_SPF)
                 ),
                ('SlimPoly', SlimPoly(**_kwargs)),
                ('MLR', LinearRegression(fit_intercept = True, n_jobs = 1))
            ]
        )

        check_pipeline(pipe)

        pipe.fit(_X, _y)

        _coef_pipe = pipe.steps[3][1].coef_

        # END pipe ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


        # separate ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        encoded_X = OneHotEncoder(sparse_output=False).fit_transform(_X)
        reformatted_X = _convert_format_before_SPF(encoded_X)
        deduplicated_X = SlimPoly(**_kwargs).fit_transform(reformatted_X)
        mlr = LinearRegression(fit_intercept = True, n_jobs = 1)

        mlr.fit(deduplicated_X, _y)

        _coef_separate = mlr.coef_

        # END separate ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        assert np.allclose(_coef_pipe, _coef_separate)







