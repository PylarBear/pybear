# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.ColumnDeduplicateTransformer._partial_fit. \
    _dupl_idxs import _dupl_idxs

import uuid
import numpy as np
import pandas as pd
import scipy.sparse as ss

import pytest


class Fixtures:

    @staticmethod
    @pytest.fixture(scope='module')
    def _rtol():
        return 1e-6


    @staticmethod
    @pytest.fixture(scope='module')
    def _atol():
        return 1e-6


    @staticmethod
    @pytest.fixture(scope='module')
    def _equal_nan():
        return False


    @staticmethod
    @pytest.fixture(scope='module')
    def _n_jobs():
        return 1     # leave set at 1 because of confliction


    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (100, 20)


    @staticmethod
    @pytest.fixture(scope='module')
    def _X_base(_X_factory, _shape):
        return _X_factory(
            _dupl=None,
            _format='np',
            _dtype='flt',
            _columns=None,
            _shape=_shape
        )


    @staticmethod
    @pytest.fixture(scope='module')
    def _init_duplicates():
        return [
            [1, 15],
            [3, 8, 12]
        ]


    @staticmethod
    @pytest.fixture(scope='module')
    def _X_initial(_X_base, _init_duplicates):
        _X_initial = _X_base.copy()
        for _set in _init_duplicates:
            for idx in _set[1:]:
                _X_initial[:, idx] = _X_initial[:, _set[0]]
        return _X_initial


    @staticmethod
    @pytest.fixture(scope='module')
    def _less_duplicates():
        return [
            [1, 15],
            [3, 12]
        ]


    @staticmethod
    @pytest.fixture(scope='module')
    def _X_less_dupl_found(_X_base, _less_duplicates):
        _X_less_dupl_found = _X_base.copy()
        for _set in _less_duplicates:
            for idx in _set[1:]:
                _X_less_dupl_found[:, idx] = _X_less_dupl_found[:, _set[0]]
        return _X_less_dupl_found


    @staticmethod
    @pytest.fixture(scope='module')
    def _more_duplicates():
        return [[1, 4, 15], [3, 8, 12]]


    @staticmethod
    @pytest.fixture(scope='module')
    def _X_more_dupl_found(_X_base, _more_duplicates):
        _X_more_dupl_found = _X_base.copy()
        for _set in _more_duplicates:
            for idx in _set[1:]:
                _X_more_dupl_found[:, idx] = _X_more_dupl_found[:, _set[0]]
        return _X_more_dupl_found



class TestDuplIdxs(Fixtures):

    @pytest.mark.parametrize('_format',
        (
             'np', 'pd', 'csr_matrix', 'csc_matrix', 'coo_matrix', 'dia_matrix',
             'lil_matrix', 'dok_matrix', 'bsr_matrix', 'csr_array', 'csc_array',
             'coo_array', 'dia_array', 'lil_array', 'dok_array', 'bsr_array'
        )
    )
    def test_rejects_ss_coo_dia_bsr(
        self, _format, _shape, _X_base, _init_duplicates, _rtol, _atol,
        _equal_nan, _n_jobs
    ):

        if _format == 'np':
            _X_wip = _X_base
        elif _format == 'pd':
            _X_wip = pd.DataFrame(
                data=_X_base,
                columns=[str(uuid.uuid4)[:5] for _ in range(_shape[1])]
            )
        elif _format == 'csr_matrix':
            _X_wip = ss._csr.csr_matrix(_X_base)
        elif _format == 'csc_matrix':
            _X_wip = ss._csc.csc_matrix(_X_base)
        elif _format == 'coo_matrix':
            _X_wip = ss._coo.coo_matrix(_X_base)
        elif _format == 'dia_matrix':
            _X_wip = ss._dia.dia_matrix(_X_base)
        elif _format == 'lil_matrix':
            _X_wip = ss._lil.lil_matrix(_X_base)
        elif _format == 'dok_matrix':
            _X_wip = ss._dok.dok_matrix(_X_base)
        elif _format == 'bsr_matrix':
            _X_wip = ss._bsr.bsr_matrix(_X_base)
        elif _format == 'csr_array':
            _X_wip = ss._csr.csr_array(_X_base)
        elif _format == 'csc_array':
            _X_wip = ss._csc.csc_array(_X_base)
        elif _format == 'coo_array':
            _X_wip = ss._coo.coo_array(_X_base)
        elif _format == 'dia_array':
            _X_wip = ss._dia.dia_array(_X_base)
        elif _format == 'lil_array':
            _X_wip = ss._lil.lil_array(_X_base)
        elif _format == 'dok_array':
            _X_wip = ss._dok.dok_array(_X_base)
        elif _format == 'bsr_array':
            _X_wip = ss._bsr.bsr_array(_X_base)
        else:
            raise Exception

        if isinstance(_X_wip,
            (ss.coo_matrix, ss.coo_array, ss.dia_matrix,
            ss.dia_array, ss.bsr_matrix, ss.bsr_array)
        ):
            with pytest.raises(AssertionError):
                _dupl_idxs(_X_wip, None, _rtol, _atol, _equal_nan, _n_jobs)
        else:
            out = _dupl_idxs(_X_wip, None, _rtol, _atol, _equal_nan, _n_jobs)
            assert isinstance(out, list)


    def test_first_pass(
        self, _X_initial, _init_duplicates, _rtol, _atol, _equal_nan, _n_jobs
    ):

        # on first pass, the output of _find_duplicates is returned directly.
        # _find_duplicates is tested elsewhere for all input types. Only need
        # to test with numpy arrays here.

        out = _dupl_idxs(
            _X_initial, None, _rtol, _atol, _equal_nan, _n_jobs
        )

        for idx in range(len(out)):
            #                               vvvvvvvvvvvvvvvvvvvvv
            assert np.array_equiv(out[idx], _init_duplicates[idx])


    def test_less_dupl_found(
        self, _X_less_dupl_found, _init_duplicates, _less_duplicates, _rtol,
        _atol, _equal_nan, _n_jobs
    ):

        # on a partial fit where less duplicates are found, outputted melded
        # duplicates should reflect the lesser columns

        out = _dupl_idxs(
            _X_less_dupl_found, _init_duplicates, _rtol, _atol, _equal_nan,
            _n_jobs
        )

        for idx in range(len(out)):
            #                               vvvvvvvvvvvvvvvvvvvvv
            assert np.array_equiv(out[idx], _less_duplicates[idx])


    def test_more_dupl_found(
        self, _X_more_dupl_found, _init_duplicates, _more_duplicates, _rtol,
        _atol, _equal_nan, _n_jobs
    ):

        # on a partial fit where more duplicates are found, outputted melded
        # duplicates should not add the newly found columns

        out = _dupl_idxs(
            _X_more_dupl_found, _init_duplicates, _rtol, _atol, _equal_nan,
            _n_jobs
        )

        for idx in range(len(out)):
            #                               vvvvvvvvvvvvvvvvvvvvv
            assert np.array_equiv(out[idx], _init_duplicates[idx])


    def test_more_and_less_duplicates_found(
        self, _init_duplicates, _less_duplicates, _more_duplicates, _X_initial,
        _X_more_dupl_found, _X_less_dupl_found, _rtol, _atol, _equal_nan,
        _n_jobs
    ):

        duplicates_ = _dupl_idxs(
            _X_initial, None, _rtol, _atol, _equal_nan, _n_jobs
        )

        duplicates_ = _dupl_idxs(
            _X_more_dupl_found, duplicates_, _rtol, _atol, _equal_nan, _n_jobs
        )

        duplicates_ = _dupl_idxs(
            _X_less_dupl_found, duplicates_, _rtol, _atol, _equal_nan, _n_jobs
        )

        # _less_duplicates must be the correct output
        for idx in range(len(duplicates_)):
            assert np.array_equiv(duplicates_[idx], _less_duplicates[idx])



    def test_no_duplicates_found(
        self, _X_initial, _X_base, _init_duplicates, _rtol, _atol, _equal_nan,
            _n_jobs
    ):

        duplicates_ = _dupl_idxs(
            _X_base, None, _rtol, _atol, _equal_nan, _n_jobs
        )

        assert duplicates_ == []

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        duplicates_ = _dupl_idxs(
            _X_initial, None, _rtol, _atol, _equal_nan, _n_jobs
        )

        duplicates_ = _dupl_idxs(
            _X_base, duplicates_, _rtol, _atol, _equal_nan, _n_jobs
        )

        assert duplicates_ == []



    def test_special_case_accuracy(
        self, _X_factory, _rtol, _atol, _equal_nan, _n_jobs, _shape
    ):

        # test cases where columns repeat, but in different groups

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        _fst_duplicates = [[0, 1, 2], [3, 4, 5]]
        _scd_duplicates = [[0, 4, 5], [1, 2, 3]]

        _scd_X = _X_factory(
            _dupl=_scd_duplicates,
            _has_nan=False,
            _format='np',
            _dtype='flt',
            _columns=None,
            _zeros=None,
            _shape=_shape
        )

        out = _dupl_idxs(
            _scd_X, _fst_duplicates, _rtol, _atol, _equal_nan, _n_jobs
        )

        assert out == [[1, 2], [4, 5]]

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        _fst_duplicates = [[1, 3, 5], [0, 2, 4]]
        _scd_duplicates = [[0, 2, 4], [1, 3, 5]]

        _scd_X = _X_factory(
            _dupl=_scd_duplicates,
            _has_nan=False,
            _format='np',
            _dtype='flt',
            _columns=None,
            _zeros=None,
            _shape=_shape
        )

        out = _dupl_idxs(
            _scd_X, _fst_duplicates, _rtol, _atol, _equal_nan, _n_jobs
        )

        assert out == [[0, 2, 4], [1, 3, 5]]

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        _fst_duplicates = [[0, 1], [2, 3], [4, 5]]
        _scd_duplicates = [[0, 4], [1, 3], [2, 5]]

        _scd_X = _X_factory(
            _dupl=_scd_duplicates,
            _has_nan=False,
            _format='np',
            _dtype='flt',
            _columns=None,
            _zeros=None,
            _shape=_shape
        )

        out = _dupl_idxs(
            _scd_X, _fst_duplicates, _rtol, _atol, _equal_nan, _n_jobs
        )

        assert out == []






