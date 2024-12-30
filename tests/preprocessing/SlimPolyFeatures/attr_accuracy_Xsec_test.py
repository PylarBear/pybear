# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.SlimPolyFeatures.SlimPolyFeatures import \
    SlimPolyFeatures as SlimPoly

import itertools
import numpy as np
import pandas as pd
import scipy.sparse as ss

import pytest





class TestAttrAccuracy:


    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (9, 3)   # DO NOT CHANGE THIS!!!!


    @staticmethod
    @pytest.fixture(scope='function')
    def _kwargs():

        return {
            'degree': 2,
            'min_degree': 1,
            'interaction_only': True,
            'scan_X': False,
            'keep': 'first',
            'sparse_output': True,
            'feature_name_combiner': 'as_indices',
            'equal_nan': True,
            'rtol': 1e-5,
            'atol': 1e-8,
            'n_jobs': 1
        }


    @staticmethod
    @pytest.fixture(scope='module')
    def _columns(_master_columns, _shape):
        return _master_columns.copy()[:_shape[1]]


    @pytest.mark.parametrize('X_format', ('np', 'pd', 'csr_array'))
    @pytest.mark.parametrize('min_degree', (1, 2))
    @pytest.mark.parametrize('X_is_rigged', (True, False))
    @pytest.mark.parametrize('intx_only', (True, False))
    @pytest.mark.parametrize('sparse_output', (True, False))
    def test_attr_accuracy(
        self, _X_factory, _master_columns, _columns, _kwargs, _shape,
        X_format, min_degree, X_is_rigged, intx_only, sparse_output
    ):

        _kwargs['min_degree'] = min_degree
        _kwargs['interaction_only'] = intx_only

        # BUILD X v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v
        if X_is_rigged:
            # with this rigging, all squared columns != original column.
            # all interactions equal the same thing, so there should be
            # only one feature in poly.
            _X = np.array(
                [
                    [2, 0, 0],
                    [2, 0, 0],
                    [2, 0, 0],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [0, 0, 2],
                    [0, 2, 0],
                    [0, 2, 0]
                ],
                dtype=np.uint8
            )
        else:
            _X = _X_factory(
                _dupl=None,
                _format='np',   # <+===== SET THE ACTUAL AFTER THIS
                _dtype='flt',
                _has_nan=False,
                _constants=None,
                _columns=_columns,
                _zeros=None,
                _shape=_shape
            )


        if X_format == 'np':
            pass
        elif X_format == 'pd':
            _X = pd.DataFrame(
                data=_X,
                columns=_columns
            )
        elif X_format == 'csr_matrix':
            _X = ss._csr.csr_matrix(_X)
        elif X_format == 'csc_matrix':
            _X = ss._csc.csc_matrix(_X)
        elif X_format == 'coo_matrix':
            _X = ss._coo.coo_matrix(_X)
        elif X_format == 'dia_matrix':
            _X = ss._dia.dia_matrix(_X)
        elif X_format == 'lil_matrix':
            _X = ss._lil.lil_matrix(_X)
        elif X_format == 'dok_matrix':
            _X = ss._dok.dok_matrix(_X)
        elif X_format == 'bsr_matrix':
            _X = ss._bsr.bsr_matrix(_X)
        elif X_format == 'csr_array':
            _X = ss._csr.csr_array(_X)
        elif X_format == 'csc_array':
            _X = ss._csc.csc_array(_X)
        elif X_format == 'coo_array':
            _X = ss._coo.coo_array(_X)
        elif X_format == 'dia_array':
            _X = ss._dia.dia_array(_X)
        elif X_format == 'lil_array':
            _X = ss._lil.lil_array(_X)
        elif X_format == 'dok_array':
            _X = ss._dok.dok_array(_X)
        elif X_format == 'bsr_array':
            _X = ss._bsr.bsr_array(_X)
        else:
            raise Exception
        # END BUILD X v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^

        TestCls = SlimPoly(**_kwargs)

        # must be fitted to access all of these attrs & properties!
        if X_format in ('coo', 'bsr', 'dia'):
            # warns because SPF is making a copy of X because not indexable
            with pytest.warns():
                TestCls.fit(_X)
        else:
            assert TestCls.fit(_X) is TestCls

        # pizza unquote this
        """
        # THE SIMPLE TESTS v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
        # get feature_names_out - - - - - - - - - - - - - -
        # the lion's share of GFNO test is handled in get_feature_names_out_accuracy
        assert isinstance(TestCls.get_feature_names_out(), np.ndarray)
        # - - - - - - - - - - - - - - - - - - - - -

        # sparse_output - - - - - - - - - - - - - -
        if _kwargs['sparse_output'] is True:
            assert isinstance(
                TestCls.transform(_X), (ss.csr_matrix, ss.csr_array)
            )
        elif _kwargs['sparse_output'] is False:
            assert isinstance(TestCls.transform(_X), type(_X))
        # END sparse_output - - - - - - - - - - - - - -

        # feature_names_in_ - - - - - - - - - - - - - - -
        if X_format == 'pd':
            _fni = TestCls.feature_names_in_
            assert isinstance(_fni, np.ndarray)
            assert _fni.dtype == object
            assert len(_fni) == _X.shape[1]
            assert np.array_equiv(_fni, _columns), \
                f"{_fni} after fit() != originally passed columns"
        else:
            with pytest.raises(AttributeError):
                TestCls.feature_names_in_
        # END feature_names_in_ - - - - - - - - - - - - - - -

        # n_features_in_ - - - - - - - - - - - - -
        _nfi = TestCls.n_features_in_
        assert isinstance(_nfi, int)
        assert _nfi == _shape[1]
        # - - - - - - - - - - - - - - - - - - - - -

        # get_params - - - - - - - - - - - - - - - -
        _params = TestCls.get_params()
        assert isinstance(_params, dict)
        for _param, _value in _params.items():
            assert _kwargs[_param] == _value
        # - - - - - - - - - - - - - - - - - - - - -

        # set_params - - - - - - - - - - - - - - - -
        # remember most are blocked once fit!
        # the lion's share of test is handled in set_params_test
        TestCls.set_params(sparse_output=True, n_jobs=2)
        assert TestCls.sparse_output is True
        assert TestCls.n_jobs == 2
        # - - - - - - - - - - - - - - - - - - - - -
        # END THE SIMPLE TESTS v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
        """

        # expansion_combinations_ - - - - - - - - - - - - - - - - - - - - - - -
        out = TestCls.expansion_combinations_
        assert isinstance(out, tuple)
        assert all(map(isinstance, out, (tuple for _ in out)))

        # first order expansion.... always the same, regardless of intx_only
        _first_order_expansion = list(itertools.combinations(range(_shape[1]), 1))
        _scd_order_expansion_intx_only = list(itertools.combinations(range(_shape[1]), 2))
        _scd_order_expansion_not_intx_only = list(itertools.combinations_with_replacement(range(_shape[1]), 2))
        _rigged_scd_order_expansion_intx_only = [(0, 1)]
        _rigged_scd_order_expansion_not_intx_only = [(0, 0), (0, 1), (1, 1), (2, 2)]

        if X_is_rigged:
            # all squared columns != original column.
            # all interactions equal the same thing, so there should be
            # only one feature in poly.
            if min_degree == 1:
                if intx_only:
                    assert out == tuple(_first_order_expansion +_rigged_scd_order_expansion_intx_only)
                elif not intx_only:
                    assert out == tuple(_first_order_expansion + _rigged_scd_order_expansion_not_intx_only)
            elif min_degree == 2:
                if intx_only:
                    assert out == tuple(_rigged_scd_order_expansion_intx_only)
                elif not intx_only:
                    assert out == tuple(_rigged_scd_order_expansion_not_intx_only)

        elif not X_is_rigged:
            if min_degree == 1:
                if intx_only:
                    assert out == tuple(_first_order_expansion +_scd_order_expansion_intx_only)
                elif not intx_only:
                    assert out == tuple(_first_order_expansion + _scd_order_expansion_not_intx_only)
            elif min_degree == 2:
                if intx_only:
                    assert out == tuple(_scd_order_expansion_intx_only)
                elif not intx_only:
                    assert out == tuple(_scd_order_expansion_not_intx_only)
        # END expansion_combinations_ - - - - - - - - - - - - - - - - - - - - -


        """
        # poly_duplicates_
        assert isinstance(TestCls.poly_duplicates_, list)

        # kept_poly_duplicates_
        assert isinstance(TestCls.kept_poly_duplicates_, dict)

        # dropped_poly_duplicates_
        assert isinstance(TestCls.dropped_poly_duplicates_, dict)
        """

        # pizza build another test case where this will actually have values
        # poly_constants_ - - - - - - - - - - - - - - - - - - - - - - -
        out = TestCls.poly_constants_
        assert isinstance(out, dict)
        # should be empty dict in every case for these specific tests
        assert len(out) == 0
        # END poly_constants_ - - - - - - - - - - - - - - - - - - - - -




