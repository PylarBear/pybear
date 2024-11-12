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
    @pytest.mark.parametrize('_has_nan', (False, 1, 3, 5, 9)) # use numbers, need exact
    def test_accuracy_np_pd(
        self, _X_factory, _master_columns, _shape, X_format, X_dtype, _has_nan
    ):

        # by using nan_mask on ('flt', 'int', 'str', 'obj', 'hybrid'), both
        # nan_mask_numerical and nan_mask_string are tested

        _X = _X_factory(
            _dupl=None,
            _format=X_format,
            _dtype=X_dtype,
            _has_nan=_has_nan,
            _columns=_master_columns[:_shape[1]] if X_format == 'pd' else None,
            _zeros=None,
            _shape=_shape
        )

        OUT = nan_mask(_X)

        assert isinstance(OUT, np.ndarray)

        for _col_idx in range(OUT.shape[1]):

            measured_num_nans = np.sum(OUT[:, _col_idx])

            if _has_nan is False:
                assert measured_num_nans == 0
            else:
                assert measured_num_nans == _has_nan



    @pytest.mark.parametrize('X_format',
        (
        'csr_matrix', 'csc_matrix', 'coo_matrix', 'dia_matrix', 'lil_matrix',
        'bsr_matrix', 'dok_matrix', 'csr_array', 'csc_array', 'coo_array',
        'dia_array', 'lil_array', 'bsr_array', 'dok_array'
        )
    )
    @pytest.mark.parametrize('X_dtype', ('flt', 'int')) # ss can only take num
    @pytest.mark.parametrize('_has_nan', (False, 1, 3, 5, 9)) #use numbers, need exact
    def test_accuracy_scipy(
        self, _X_factory, _master_columns, _shape, X_format, X_dtype, _has_nan
    ):

        # 'dok' is the only ss that doesnt have a 'data' attribute, and therefore
        # isnt handled by nan_masking(). 'lil' cant be masked in an elegant way, so
        # also is not handled by nan_masking(). all other ss can only take numeric.
        # by using nan_mask on ('flt', 'int'), only nan_mask_numerical is tested

        _X = _X_factory(
            _dupl=None,
            _format=X_format,
            _dtype=X_dtype,
            _has_nan=_has_nan,
            _columns=None,
            _zeros=None,
            _shape=_shape
        )

        # use nan_mask on the ss if it isnt dok
        if 'dok' in X_format or 'lil' in X_format:
            with pytest.raises(TypeError):
                nan_mask(_X)
            pytest.skip(reason=f"unable to do any tests if dok or lil")
        else:
            out = nan_mask(_X)

        assert isinstance(out, np.ndarray)

        # get the original numpy array format of _X
        _X_as_np = _X.toarray()
        # we proved in the first test that nan_mask works correctly on
        # np, so use that on _X_as_np to get a referee
        _ref_out = nan_mask(_X_as_np)

        # if we use the ss nan_mask to set the ss nan values to some other
        # actual number, then if we use toarray() to convert to np, then
        # the locations of the rigged numbers should match the nan_mask
        # on _X_as_np

        _X.data[out] = -99

        _X = _X.toarray()

        assert np.all(_X[_ref_out] == -99)











