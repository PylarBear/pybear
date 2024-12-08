# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.NoDupPolyFeatures._validation._validation \
    import _validation

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



    @pytest.mark.parametrize('_degree', (1, 2))
    @pytest.mark.parametrize('_min_degree', (0, 1)) # 1))
    @pytest.mark.parametrize('_keep', ('first', 'last', 'random'))
    @pytest.mark.parametrize('_interaction_only', (True, False))
    @pytest.mark.parametrize('_sparse_output', (True, False))
    @pytest.mark.parametrize('_rtol', (1e-6, )) # 1e-1))
    @pytest.mark.parametrize('_atol', (1e-6, )) # 1))
    @pytest.mark.parametrize('_scan_X', (True, False))
    @pytest.mark.parametrize('_equal_nan', (True, False))
    @pytest.mark.parametrize('_n_jobs', (-1, 1))
    def test_accepts_good(
        self, _X, _columns, _degree, _min_degree, _keep, _interaction_only,
        _sparse_output, _rtol, _atol, _scan_X, _equal_nan, _n_jobs, _shape
    ):

        _validation(
            _X,
            _columns,
            _degree,
            _min_degree,
            _scan_X,
            _keep,
            _interaction_only,
            _sparse_output,
            _rtol,
            _atol,
            _equal_nan,
            _n_jobs
        )










