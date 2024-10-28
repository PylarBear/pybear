# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.utilities._nan_masking import nan_mask

import numpy as np


import pytest





class TestNanMasking:

    # tests using _X_factory. _X_factory is a fixture that can introduce
    # into X a controlled amount of nan-like representations.

    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (100, 10)



    @pytest.mark.parametrize('X_format', ('np', 'pd'))
    @pytest.mark.parametrize('X_dtype', ('flt', 'int', 'str', 'obj', 'hybrid'))
    @pytest.mark.parametrize('_has_nan', (False, 1, 3, 5, 9)) # dont use True, need exact
    def test_accuracy(
        self, _X_factory, _master_columns, _shape, X_format, X_dtype, _has_nan
    ):

        _X = _X_factory(
            _dupl=None,
            _format=X_format,
            _dtype=X_dtype,
            _has_nan=_has_nan,
            _columns=_master_columns[:_shape[1]] if X_format == 'pd' else None,
            _zeros=None,
            _shape=_shape
        )


        # by using nan_mask on ('flt', 'int', 'str', 'obj', 'hybrid'), both
        # nan_mask_numerical and nan_mask_string are tested
        OUT = nan_mask(_X)

        assert isinstance(OUT, np.ndarray)

        for _col_idx in range(OUT.shape[1]):

            measured_num_nans = np.sum(OUT[:, _col_idx])

            if _has_nan is False:
                assert measured_num_nans == 0
            else:
                assert measured_num_nans == _has_nan




