# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#
from pybear.utilities import nan_mask_numerical
from pybear.utilities._nan_masking import nan_mask

import numpy as np
import polars as pl

import pytest





class TestNanMasking:

    # by using nan_mask on ('flt', 'int', 'str', 'obj', 'hybrid'), both
    # nan_mask_numerical and nan_mask_string are tested

    # tests using _X_factory. _X_factory is a fixture that can introduce
    # into X a controlled amount of nan-like representations.

    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (100, 10)


    @pytest.mark.parametrize('_dim', (1, 2))
    @pytest.mark.parametrize('_container', (list, tuple, set))
    @pytest.mark.parametrize('X_dtype', ('flt', 'int', 'str', 'obj', 'hybrid'))
    @pytest.mark.parametrize('_has_nan', (False, 1, 3, 5, 9)) # use numbers, need exact
    def test_accuracy_python_builtins(
        self, _X_factory, _master_columns, _shape, _dim, _container, X_dtype, _has_nan
    ):

        # skip impossible -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        if _container is set:
            if _dim == 2 :
                pytest.skip(reason=f"cant have 2D sets")
            # can only test this when there is only one nan-like in a set!
            # otherwise, set will consolidate similar nans and change the count!
            if _has_nan > 1:
                pytest.skip(reason=f'have multiple similar nans with set')
        if X_dtype == 'hybrid' and _dim == 1:
            pytest.skip(reason=f'cant have hybrid when a single vector')

        # END skip impossible -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        _X = _X_factory(
            _dupl=None,
            _format='np',
            _dtype=X_dtype,
            _has_nan=_has_nan,
            _columns=None,
            _zeros=None,
            _shape=_shape
        )

        if _dim == 1:
            _X = _container(_X[:, 0])
        elif _dim == 2:
            _X = _container(map(_container, _X))
        else:
            raise Exception

        assert isinstance(_X, _container)

        OUT = nan_mask(_X)

        assert isinstance(OUT, np.ndarray)

        _n_columns = 1 if _dim == 1 else OUT.shape[1]
        for _col_idx in range(_n_columns):

            if _dim == 1:
                measured_num_nans = np.sum(OUT)
            else:
                measured_num_nans = np.sum(OUT[:, _col_idx])

            if _has_nan is False:
                assert measured_num_nans == 0
            else:
                assert measured_num_nans == _has_nan


    @pytest.mark.parametrize('_dim', (1, 2))
    @pytest.mark.parametrize('X_format', ('np', 'pd', 'np_masked'))
    @pytest.mark.parametrize('X_dtype', ('flt', 'int', 'str', 'obj', 'hybrid'))
    @pytest.mark.parametrize('_has_nan', (False, 1, 3, 5, 9)) # use numbers, need exact
    def test_accuracy_np_pd(
        self, _X_factory, _master_columns, _shape, _dim, X_format, X_dtype, _has_nan
    ):

        # skip impossible conditions -- -- -- -- -- -- -- -- -- -- -- -- -- --

        if X_dtype == 'hybrid' and _dim == 1:
            pytest.skip(reason=f'cant have hybrid when a single vector')

        # END skip impossible conditions -- -- -- -- -- -- -- -- -- -- -- -- --

        _format_dict = {
            'np': 'np',
            'pd': 'pd',
            'np_masked': 'np'
        }

        _X = _X_factory(
            _dupl=None,
            _format=_format_dict[X_format],
            _dtype=X_dtype,
            _has_nan=_has_nan,
            _columns=_master_columns[:_shape[1]] if X_format == 'pd' else None,
            _zeros=None,
            _shape=_shape
        )

        if _dim == 1:
            if X_format == 'np':
                _X = _X[:, 0]
            elif X_format == 'pd':
                _X = _X.iloc[:, 1].squeeze()

        if X_format == 'np_masked':
            _X = np.ma.masked_array(_X)
            with pytest.raises(TypeError):
                nan_mask(_X)
            pytest.skip(reason=f"cant do rest of tests after except")

        OUT = nan_mask(_X)

        assert isinstance(OUT, np.ndarray)

        _n_columns = 1 if _dim == 1 else OUT.shape[1]
        for _col_idx in range(_n_columns):

            if _dim == 1:
                measured_num_nans = np.sum(OUT)
            else:
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

        # use nan_mask on the ss if it isnt dok or lil
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


    @pytest.mark.parametrize('_dim', (1, 2))
    @pytest.mark.parametrize('X_dtype', ('flt', 'int', 'str', 'obj', 'hybrid'))
    @pytest.mark.parametrize('_has_nan', (False, 1, 3, 5, 9)) # use numbers, need exact
    def test_accuracy_polars(
        self, _X_factory, _master_columns, _shape, _dim, X_dtype, _has_nan
    ):

        # skip impossible conditions -- -- -- -- -- -- -- -- -- -- -- -- -- --

        if X_dtype == 'hybrid' and _dim == 1:
            pytest.skip(reason=f'cant have hybrid when a single vector')

        # END skip impossible conditions -- -- -- -- -- -- -- -- -- -- -- -- --

        _X = _X_factory(
            _dupl=None,
            _format='pd',
            _dtype=X_dtype,
            _has_nan=_has_nan,
            _columns=_master_columns[:_shape[1]],
            _zeros=None,
            _shape=_shape
        )

        if _dim == 1:
            _X = _X.iloc[:, 0].squeeze()

        # polars cant take 'nan' 'NaN' 'NAN' '<NA>' -- map these to None
        _X[_X == 'nan'] = None
        _X[_X == 'NaN'] = None
        _X[_X == 'NAN'] = None
        _X[_X == '<NA>'] = None

        _X = pl.from_pandas(
            data=_X,
            schema_overrides=None,
            rechunk=True,
            nan_to_null=True,
            include_index=False
        )

        if _dim == 1:
            assert isinstance(_X, pl.Series)

        OUT = nan_mask(_X)

        assert isinstance(OUT, np.ndarray)

        _n_columns = 1 if _dim == 1 else OUT.shape[1]
        for _col_idx in range(_n_columns):

            if _dim == 1:
                measured_num_nans = np.sum(OUT)
            elif _dim == 2:
                measured_num_nans = np.sum(OUT[:, _col_idx])

            if _has_nan is False:
                assert measured_num_nans == 0
            else:
                assert measured_num_nans == _has_nan


    def test_nan_mask_numerical_takes_str_numbers(self):

        # 1D -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        _X = list(map(str, list(range(10))))

        assert all(map(isinstance, _X, (str for _ in _X)))
        np.array(_X, dtype=np.float64)

        _X[1] = 'nan'
        _X[2] = np.nan

        ref = np.zeros((10,)).astype(bool)
        ref[1] = True
        ref[2] = True
        ref = ref.tolist()

        out = nan_mask_numerical(_X)

        assert np.array_equal(out, ref)
        # END 1D -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # 2D -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        _X = np.random.randint(0, 10, (5, 3)).astype(str)

        assert all(map(isinstance, _X[0], (str for _ in _X[0])))
        np.array(_X, dtype=np.float64)

        _X[1][0] = 'nan'
        _X[2][0] = np.nan

        ref = np.zeros((5, 3)).astype(bool)
        ref[1][0] = True
        ref[2][0] = True
        ref = ref.tolist()

        out = nan_mask_numerical(_X)

        assert np.array_equal(out, ref)

        # END 2D -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
