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
    @pytest.fixture(scope='function')
    def _X(_X_factory, _format, _shape):
        return _X_factory(_format=_format, _shape=_shape)


    @staticmethod
    @pytest.fixture(scope='module')
    def _columns(_master_columns, _shape):
        return _master_columns.copy()[:_shape[1]]


    @pytest.mark.parametrize('_format', ('np', 'pd'))
    @pytest.mark.parametrize('_keep',
        ('first', 'last', 'random', 'none', {'Intercept': 1}, 0, 'string', lambda x: 1)
    )
    @pytest.mark.parametrize('_equal_nan', (True, False))
    @pytest.mark.parametrize('_rtol', (1e-6, 1e-1))
    @pytest.mark.parametrize('_atol', (1e-6, 1))
    @pytest.mark.parametrize('_n_jobs', (None, -1, 1))
    @pytest.mark.parametrize('columns_is_passed', (True, False))
    def test_accepts_good(
        self, _X, _columns, _format, _keep, _rtol, _atol, _equal_nan, _n_jobs,
        columns_is_passed
    ):

        if columns_is_passed and _format == 'np':
            pytest.skip(reason=f"cannot pass columns to np array")

        if _keep == 'string':

            if not columns_is_passed:
                pytest.skip(reason=f"cannot have string _keep when columns are not passed")

            if _format == 'np':

                with pytest.raises(ValueError):
                    _validation(
                        _X,
                        None,
                        _keep,
                        _equal_nan,
                        _rtol,
                        _atol,
                        _n_jobs
                    )
            else:
                _keep = _columns[0]

        _validation(
            _X,
            _columns if columns_is_passed else None,
            _keep,
            _equal_nan,
            _rtol,
            _atol,
            _n_jobs
        )











