# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.ColumnDeduplicateTransformer._partial_fit. \
    _dupl_idxs import _dupl_idxs


import pytest
import numpy as np





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
        return -1


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










