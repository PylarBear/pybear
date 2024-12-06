# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.preprocessing.NoDupPolyFeatures._validation._validation \
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



    @pytest.mark.parametrize('_degree', (1, 2))
    @pytest.mark.parametrize('_min_degree', (0, 1)) # 1))
    @pytest.mark.parametrize('_drop_duplicates', (True, False))
    # @pytest.mark.parametrize('_conflict', ('raise', 'ignore'))  # pizza on the block
    @pytest.mark.parametrize('_keep', ('first', 'last', 'random'))
    @pytest.mark.parametrize('_interaction_only', (True, False))
    # @pytest.mark.parametrize('_include_bias', (True, False))  # pizza on the block
    @pytest.mark.parametrize('_drop_constants', (True, False))
    @pytest.mark.parametrize('_drop_collinear', (True, False))
    @pytest.mark.parametrize('_output_sparse', (True, False))
    # @pytest.mark.parametrize('_order', ('C', 'F'))  # pizza on the block
    @pytest.mark.parametrize('_rtol', (1e-6, )) # 1e-1))
    @pytest.mark.parametrize('_atol', (1e-6, )) # 1))
    @pytest.mark.parametrize('_equal_nan', (True, False))
    @pytest.mark.parametrize('_n_jobs', (-1, 1))
    def test_accepts_good(
        self, _X, _columns, _degree, _min_degree, _drop_duplicates, #_conflict, _include_bias, _do_not_drop, _order,
        _keep, _interaction_only, _drop_constants, _drop_collinear,
        _output_sparse, _rtol, _atol, _equal_nan, _n_jobs, _shape
    ):

        _validation(
            _X,
            _columns,
            _degree,
            _min_degree,
            _drop_duplicates,
            _keep,
            # _do_not_drop,  # pizza on the block
            # _conflict,  # pizza on the block
            _interaction_only,
            # _include_bias,  # pizza on the block
            _drop_constants,
            _drop_collinear,
            _output_sparse,
            # _order,  # pizza on the block
            _rtol,
            _atol,
            _equal_nan,
            _n_jobs
        )




