# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.preprocessing.InterceptManager._validation._validation import (
    _validation
)

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

    # pizza this isnt being used
    @staticmethod
    @pytest.fixture(scope='module')
    def _columns(_master_columns, _shape):
        return _master_columns.copy()[:_shape[1]]


    @pytest.mark.parametrize('_keep',
        ('first', 'last', 'random', None, {'Intercept': 1})
    )
    @pytest.mark.parametrize('_equal_nan', (True, False))
    @pytest.mark.parametrize('_rtol', (1e-6, 1e-1))
    @pytest.mark.parametrize('_atol', (1e-6, 1))
    @pytest.mark.parametrize('_n_jobs', (None, -1, 1))
    def test_accepts_good(
        self, _X, _keep, _rtol, _atol, _equal_nan, _n_jobs
    ):


        _validation(
            _X,
            _keep,
            _equal_nan,
            _rtol,
            _atol,
            _n_jobs
        )











