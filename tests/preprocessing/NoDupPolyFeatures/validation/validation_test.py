# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.preprocessing.ColumnDeduplicateTransformer._validation._validation \
    import _validation

import numpy as np

import pytest



class TestValidation:

    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (50, 10)

    @staticmethod
    @pytest.fixture(scope='module')
    def _X(_X_factory, _shape):
        return _X_factory(_format='np', _shape=_shape)

    @staticmethod
    @pytest.fixture(scope='module')
    def _columns(_master_columns, _shape):
        return _master_columns.copy()[:_shape[1]]

    @staticmethod
    @pytest.fixture(scope='module')
    def _do_not_drop(_shape):
        _cols = _shape[1]
        return list(np.random.choice(range(_cols), _cols//10, replace=False))


    @pytest.mark.parametrize('_conflict', ('raise', 'ignore'))
    @pytest.mark.parametrize('_keep', ('first', 'last', 'random'))
    @pytest.mark.parametrize('_rtol', (1e-6, 1e-1))
    @pytest.mark.parametrize('_atol', (1e-6, 1))
    @pytest.mark.parametrize('_equal_nan', (True, False))
    @pytest.mark.parametrize('_n_jobs', (None, -1, 1))
    def test_accepts_good(
        self, _X, _columns, _conflict, _do_not_drop, _keep, _rtol, _atol,
        _equal_nan, _n_jobs, _shape
    ):


        _validation(
            _X,
            _columns,
            _conflict,
            _do_not_drop,
            _keep,
            _rtol,
            _atol,
            _equal_nan,
            _n_jobs
        )





