# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing import MinCountTransformer as MCT
from pybear.preprocessing import ColumnDeduplicateTransformer as CDT
from pybear.utilities import check_pipeline

import uuid
import numpy as np
import pandas as pd
import scipy.sparse as ss

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    FunctionTransformer
)

import pytest


# DO THIS WITHOUT ALTERING THE 0 AXIS, ALTER AXIS 1 ONLY
# CANT PASS Y TO THE PIPE, MCT WILL RETURN IT, SO THIS PRECLUDES USING
# AN ESTIMATOR THAT NEEDS Y. DO TWO TRANSFORMERS WITH OneHotEncoder.


class TestPipeline:

    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (100, 4)


    @staticmethod
    @pytest.fixture(scope='module')
    def _kwargs():
        return {
            'count_threshold': 6,
            'ignore_float_columns': True,
            'ignore_non_binary_integer_columns': True,
            'ignore_columns': None,
            'ignore_nan': True,
            'handle_as_bool': None,
            'delete_axis_0': False,
            'reject_unseen_values': False,
            'max_recursions': 1,
            'n_jobs': None
        }


    @staticmethod
    @pytest.fixture(scope='function')
    def _X_np(_shape, _kwargs):

        # need some columns that wont be row-chopped and some
        # columns of constants that will be chopped

        _X = np.empty((_shape[0], 0))
        for c_idx in range(_shape[1]):
            if c_idx % 2 == 0:
                _X = np.hstack((
                    _X,
                    np.full((_shape[0], 1), fill_value= np.random.randint(1, 10))
                ))
            else:
                # build a vector with all values > thresh
                while True:
                    _rigged_vector = \
                        np.random.randint(
                            0,
                            _kwargs['count_threshold'] // 2,
                            (_shape[0], 1)
                        )

                    CTS = np.unique(_rigged_vector, return_counts=True)[1]

                    if min(CTS) >= _kwargs['count_threshold']:
                        del CTS
                        break

                _X = np.hstack((
                    _X,
                    _rigged_vector
                ))

        assert _X.shape == _shape

        return _X


    @staticmethod
    @pytest.fixture(scope='module')
    def _columns(_shape):
        return [str(uuid.uuid4())[:5] for _ in range(_shape[1])]


    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @pytest.mark.parametrize('_format', ('np', 'pd', 'csr', 'csc', 'bsr'))
    def test_accuracy_in_pipe_vs_out_of_pipe(
        self, _X_np, _shape, _columns, _kwargs, _format
    ):

        # this also incidentally tests functionality in a pipe

        # make a pipe of MCT, CDT, & OneHotEncoder
        # the X object needs to contain categorical data
        # fit the data on the pipeline and transform, get output
        # fit the data on the steps severally and transform, compare output


        # pipe ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # dont use OHE sparse_output, that only puts out CSR.

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        assert isinstance(_X_np, np.ndarray)

        if _format == 'np':
            _X = _X_np.copy()
        elif _format == 'pd':
            _X = pd.DataFrame(data=_X_np, columns=_columns)
        elif _format == 'csr':
            _X = ss.csr_array(_X_np)
        elif _format == 'csc':
            _X = ss.csc_array(_X_np)
        elif _format == 'bsr':
            _X = ss.bsr_array(_X_np)
        else:
            raise Exception

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # def _scipy_exploder(X):
        #     if hasattr(X, 'toarray'):
        #         return X.toarray()
        #     else:
        #         return X


        pipe = Pipeline(
            steps = [
                ('mct', MCT(**_kwargs)),
                ('cdt', CDT(keep='first', equal_nan=True, n_jobs=1)),
                ('ft',
                    FunctionTransformer(
                        lambda X: X.toarray() if hasattr(X, 'toarray') else X,
                        accept_sparse=True
                    )
                ),
                ('onehot', OneHotEncoder(sparse_output=False))
            ]
        )

        check_pipeline(pipe)

        pipe.fit(_X)

        TRFM_X_PIPE = pipe.transform(_X)

        # END pipe ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


        # separate ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        _chopped_X = MCT(**_kwargs).fit_transform(_X)
        assert isinstance(_chopped_X, type(_X))
        _CDT = CDT(keep='first', equal_nan=True, n_jobs=1)
        _CDT.set_output('default')
        _dedupl_X = _CDT.fit_transform(_chopped_X)
        TRFM_X_NOT_PIPE = \
            OneHotEncoder(sparse_output=False).fit_transform(_dedupl_X)

        # END separate ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        assert np.array_equal(TRFM_X_PIPE, TRFM_X_NOT_PIPE, equal_nan=True)


















