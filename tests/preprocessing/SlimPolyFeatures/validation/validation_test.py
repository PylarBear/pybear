# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing._SlimPolyFeatures._validation._validation \
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


    @pytest.mark.parametrize('_degree', (3, 4))
    @pytest.mark.parametrize('_min_degree', (2, 3))
    @pytest.mark.parametrize('_keep', ('first', 'last', 'random'))
    @pytest.mark.parametrize('_interaction_only', (True, False))
    @pytest.mark.parametrize('_sparse_output', (True, False))
    @pytest.mark.parametrize('_feature_name_combiner',
        (lambda _columns, x: 'strings', lambda _columns, x: 'wires'))
    @pytest.mark.parametrize('_rtol', (1e-6, )) # 1e-1))
    @pytest.mark.parametrize('_atol', (1e-6, )) # 1))
    @pytest.mark.parametrize('_scan_X', (True, False))
    @pytest.mark.parametrize('_equal_nan', (True, False))
    @pytest.mark.parametrize('_n_jobs', (-1, 1))
    def test_accepts_good(
        self, _X, _degree, _min_degree, _keep, _interaction_only, _sparse_output,
        _feature_name_combiner, _rtol, _atol, _scan_X, _equal_nan, _n_jobs, _shape
    ):

        _validation(
            _X,
            _degree,
            _min_degree,
            _scan_X,
            _keep,
            _interaction_only,
            _sparse_output,
            _feature_name_combiner,
            _rtol,
            _atol,
            _equal_nan,
            _n_jobs
        )










