# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np
import scipy.sparse as ss

from pybear.preprocessing._ColumnDeduplicateTransformer._partial_fit. \
    _dupl_idxs import _dupl_idxs



class TestDuplIdxs:


    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @staticmethod
    @pytest.fixture(scope='module')
    def _di_args():
        return {
            '_rtol': 1e-6,
            '_atol': 1e-6,
            '_equal_nan': False,
            '_n_jobs': 1     # leave set at 1 because of confliction
        }


    @staticmethod
    @pytest.fixture(scope='module')
    def _init_duplicates():
        return [[1, 6], [4, 7, 8]]


    @staticmethod
    @pytest.fixture(scope='module')
    def _less_duplicates():
        return [[1, 6], [4, 8]]


    @staticmethod
    @pytest.fixture(scope='module')
    def _more_duplicates():
        return [[1, 3, 6], [4, 7, 8]]

    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    @pytest.mark.parametrize('_format',
        (
         'np', 'pd', 'csr_matrix', 'csc_matrix', 'coo_matrix', 'dia_matrix',
         'lil_matrix', 'dok_matrix', 'bsr_matrix', 'csr_array', 'csc_array',
         'coo_array', 'dia_array', 'lil_array', 'dok_array', 'bsr_array'
        )
    )
    def test_rejects_ss_coo_dia_bsr(
        self, _X_factory, _columns, _shape, _init_duplicates, _di_args, _format
    ):

        _X_wip = _X_factory(
            _dupl=None,
            _format=_format,
            _dtype='flt',
            _has_nan=False,
            _columns=_columns,
            _shape=_shape
        )


        if isinstance(_X_wip,
            (ss.coo_matrix, ss.coo_array, ss.dia_matrix,
            ss.dia_array, ss.bsr_matrix, ss.bsr_array)
        ):
            with pytest.raises(AssertionError):
                _dupl_idxs(_X_wip, None, **_di_args)
        else:
            out = _dupl_idxs(_X_wip, None, **_di_args)
            assert isinstance(out, list)


    def test_first_pass(self, _X_factory, _init_duplicates, _shape, _di_args):

        # on first pass, the output of _find_duplicates is returned directly.
        # _find_duplicates is tested elsewhere for all input types. Only need
        # to test with numpy arrays here.

        _X_initial = _X_factory(
            _dupl=_init_duplicates, _format='np',
            _dtype='int', _columns=None, _shape=_shape
        )

        out = _dupl_idxs(_X_initial, None, **_di_args)

        for idx in range(len(out)):
            #                               vvvvvvvvvvvvvvvvvvvvv
            assert np.array_equiv(out[idx], _init_duplicates[idx])


    def test_less_dupl_found(
        self, _X_factory, _init_duplicates, _less_duplicates, _di_args, _shape
    ):

        # on a partial fit where less duplicates are found, outputted melded
        # duplicates should reflect the lesser columns

        _X_less_dupl_found = _X_factory(
            _dupl=_less_duplicates,_format='np',
            _dtype='int', _columns=None, _shape=_shape
        )

        out = _dupl_idxs(_X_less_dupl_found, _init_duplicates, **_di_args)

        for idx in range(len(out)):
            #                               vvvvvvvvvvvvvvvvvvvvv
            assert np.array_equiv(out[idx], _less_duplicates[idx])


    def test_more_dupl_found(
        self, _X_factory, _more_duplicates, _init_duplicates, _di_args, _shape
    ):

        # on a partial fit where more duplicates are found, outputted melded
        # duplicates should not add the newly found columns

        _X_more_dupl_found = _X_factory(
            _dupl=_more_duplicates,_format='np',
            _dtype='int', _columns=None, _shape=_shape
        )

        out = _dupl_idxs(_X_more_dupl_found, _init_duplicates, **_di_args)

        for idx in range(len(out)):
            #                               vvvvvvvvvvvvvvvvvvvvv
            assert np.array_equiv(out[idx], _init_duplicates[idx])


    def test_more_and_less_duplicates_found(
        self, _X_factory, _init_duplicates, _less_duplicates,
        _more_duplicates, _di_args, _shape
    ):

        _X_initial = _X_factory(
            _dupl=_init_duplicates, _format='np',
            _dtype='int', _columns=None, _shape=_shape
        )

        duplicates_ = _dupl_idxs(_X_initial, None, **_di_args)

        _X_more_dupl_found = _X_factory(
            _dupl=_more_duplicates,_format='np',
            _dtype='int', _columns=None, _shape=_shape
        )

        duplicates_ = _dupl_idxs(_X_more_dupl_found, duplicates_, **_di_args)

        _X_less_dupl_found = _X_factory(
            _dupl=_less_duplicates,_format='np',
            _dtype='int', _columns=None, _shape=_shape
        )

        duplicates_ = _dupl_idxs(_X_less_dupl_found, duplicates_, **_di_args)

        # _less_duplicates must be the correct output
        for idx in range(len(duplicates_)):
            assert np.array_equiv(duplicates_[idx], _less_duplicates[idx])


    def test_no_duplicates_found(
        self, _X_factory, _init_duplicates, _di_args, _shape
    ):

        _X_base = _X_factory(
            _dupl=None, _format='np', _dtype='int', _columns=None, _shape=_shape
        )

        duplicates_ = _dupl_idxs(_X_base, None, **_di_args)

        assert duplicates_ == []

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        _X_initial = _X_factory(
            _dupl=_init_duplicates, _format='np',
            _dtype='int', _columns=None, _shape=_shape
        )

        duplicates_ = _dupl_idxs(_X_initial, None, **_di_args)

        _X_base = _X_factory(
            _dupl=None, _format='np', _dtype='int', _columns=None, _shape=_shape
        )

        duplicates_ = _dupl_idxs(_X_base, duplicates_, **_di_args)

        assert duplicates_ == []


    def test_special_case_accuracy(self, _X_factory, _shape, _di_args):

        # test cases where columns repeat, but in different groups

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        _fst_duplicates = [[0, 1, 2], [3, 4, 5]]
        _scd_duplicates = [[0, 4, 5], [1, 2, 3]]

        _scd_X = _X_factory(
            _dupl=_scd_duplicates, _has_nan=False, _format='np',
            _dtype='flt', _columns=None, _zeros=None, _shape=_shape
        )

        out = _dupl_idxs(_scd_X, _fst_duplicates, **_di_args)

        assert out == [[1, 2], [4, 5]]

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        _fst_duplicates = [[1, 3, 5], [0, 2, 4]]
        _scd_duplicates = [[0, 2, 4], [1, 3, 5]]

        _scd_X = _X_factory(
            _dupl=_scd_duplicates, _has_nan=False, _format='np',
            _dtype='flt', _columns=None, _zeros=None, _shape=_shape
        )

        out = _dupl_idxs(_scd_X, _fst_duplicates, **_di_args)

        assert out == [[0, 2, 4], [1, 3, 5]]

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        _fst_duplicates = [[0, 1], [2, 3], [4, 5]]
        _scd_duplicates = [[0, 4], [1, 3], [2, 5]]

        _scd_X = _X_factory(
            _dupl=_scd_duplicates, _has_nan=False, _format='np',
            _dtype='flt', _columns=None, _zeros=None, _shape=_shape
        )

        out = _dupl_idxs(_scd_X, _fst_duplicates, **_di_args)

        assert out == []






