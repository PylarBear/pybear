# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from pybear.preprocessing import NanStandardizer as NS
from pybear.utilities import check_pipeline



class TestPipeline:

    @staticmethod
    @pytest.fixture(scope='function')
    def _X():

        _rows = 10
        _cols = 3

        _X = np.random.randint(0, 3, (_rows, _cols)).astype(np.float64)

        for n in range(6):

            # np.ndarray is coercing both pd.NA and None to np.nan upon
            # insertion into the array. just test np.nan and replace
            # all of them with 1.

            np_rand_row = np.random.randint(0, _rows)
            np_rand_col = np.random.randint(0, _cols)

            _X[np_rand_row, np_rand_col] = np.nan

        return _X

    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    def test_accuracy_in_pipe_vs_out_of_pipe(self, _X):

        # this also incidentally tests functionality in a pipe

        # make a pipe of NS and OneHotEncoder
        # pepper an array with np.nan and replace those with a number
        # replace with a number to simplify the array_equal process
        # fit the data on the pipeline, get the returned object
        # fit the data on the steps severally, compare both returned objects

        # pipe ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
        pipe = Pipeline(
            steps = [
                ('ns', NS(new_value=np.float64(1))),
                ('onehot', OneHotEncoder(sparse_output=False))
            ]
        )

        check_pipeline(pipe)

        pipe_X = pipe.fit_transform(_X)

        # END pipe ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


        # separate ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

        standardized_X = NS(new_value=np.float64(1)).fit_transform(_X)
        ohe_X = OneHotEncoder(sparse_output=False).fit_transform(standardized_X)
        # prove out that OHE is actually doing something
        assert 0 < standardized_X.shape[0] == ohe_X.shape[0]
        assert 0 < standardized_X.shape[1] < ohe_X.shape[1]
        # END prove out that OHE is actually doing something

        # END separate ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        assert np.array_equal(pipe_X, ohe_X)





