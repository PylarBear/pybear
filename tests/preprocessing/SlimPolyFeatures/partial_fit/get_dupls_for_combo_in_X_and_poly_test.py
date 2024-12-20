# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.SlimPolyFeatures._partial_fit.\
    _get_dupls_for_combo_in_X_and_poly import _get_dupls_for_combo_in_X_and_poly

import numpy as np
import pandas as pd
import scipy.sparse as ss
import itertools

from sklearn.preprocessing import OneHotEncoder

import pytest




class Fixtures:


    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (6, 5)


    @staticmethod
    @pytest.fixture(scope='module')
    def _good_equal_nan():
        return True


    @staticmethod
    @pytest.fixture(scope='module')
    def _rtol_atol():
        return (1e-5, 1e-8)


    @staticmethod
    @pytest.fixture(scope='module')
    def _n_jobs():
        return 1       # leave this at 1 because of contention


    @staticmethod
    @pytest.fixture(scope='module')
    def _X_ss(_X_factory, _shape):

        # pizza! X must always be ss because SPF._partial_fit() sets this before _get_dupls!

        return _X_factory(
            _dupl=None,
            _has_nan=False,
            _format='csc',
            _dtype='flt',
            _shape=_shape
        )


    @staticmethod
    @pytest.fixture(scope='module')
    def _combos(_shape):
        return list(itertools.combinations_with_replacement(range(_shape[1]), 2))


    @staticmethod
    @pytest.fixture(scope='module')
    def _good_COLUMN(_X_ss, _combos):
        # must be ndarray!
        # make _COLUMN from the last combo in _combos
        # when building _good_CSC_ARRAY fixture, build it with all the combos
        # except the last one.
        return _X_ss[:, _combos[-1]].toarray().prod(1)


    @staticmethod
    @pytest.fixture(scope='module')
    def _good_POLY_CSC(_X_ss, _shape, _combos):

        # cant have any duplicates in this
        # make _COLUMN from the last combo in _combos
        # when building _good_CSC_ARRAY fixture, build it with all the combos
        # except the last one.

        _NP = _X_ss.toarray()
        _POLY = np.empty((_shape[0], 0))
        for _combo in _combos[:-1]:
            _COLUMN =  _NP[:, _combo].prod(1).reshape((-1,1))
            for _poly_idx in range(_POLY.shape[1]):
                if np.array_equal(_COLUMN, _POLY[:, _poly_idx]):
                    break
            else:
                _POLY = np.hstack((_POLY, _COLUMN))

        return ss.csc_array(_POLY)


    # END fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *



class TestDuplsForComboValidation(Fixtures):

    # def _get_dupls_for_combo_in_X_and_poly(
    #     _COLUMN: npt.NDArray[any],
    #     _X: DataType,
    #     _POLY_CSC: ss.csc_array,
    #     _equal_nan: bool,
    #     _rtol: numbers.Real,
    #     _atol: numbers.Real,
    #     _n_jobs: Union[numbers.Integral, None]
    # ) -> list[bool]:


    # _COLUMN ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('junk_COLUMN',
        (-np.e, -1, 0, 1, np.e, True, False, 'trash', [0,1], (0,1), lambda x: x)
    )
    def test_COLUMN_rejects_junk(
        self, junk_COLUMN, _X_ss, _good_POLY_CSC, _good_equal_nan, _rtol_atol,
        _n_jobs
    ):

        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                junk_COLUMN,
                _X=_X_ss,
                _POLY_CSC=_good_POLY_CSC,
                _equal_nan=_good_equal_nan,
                _rtol=_rtol_atol[0],
                _atol=_rtol_atol[1],
                _n_jobs=_n_jobs
            )


    @pytest.mark.parametrize('bad_COLUMN', ('pd_series', 'ss_csr', 'ss_coo'))
    def test_COLUMN_rejects_bad(
        self, bad_COLUMN, _X_ss, _good_POLY_CSC, _good_equal_nan, _rtol_atol, _n_jobs
    ):

        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                bad_COLUMN,
                _X_ss,
                _good_POLY_CSC,
                _good_equal_nan,
                *_rtol_atol,
                _n_jobs=_n_jobs
            )


    # END _COLUMN ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # _X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @pytest.mark.parametrize('junk_X',
        (-np.e, -1, 0, 1, np.e, True, False, 'trash', [0,1], (0,1), lambda x: x)
    )
    def test_X_rejects_junk(
        self, _good_COLUMN, junk_X, _good_POLY_CSC, _good_equal_nan, _rtol_atol, _n_jobs
    ):

        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                _good_COLUMN,
                junk_X,
                _good_POLY_CSC,
                _good_equal_nan,
                *_rtol_atol,
                _n_jobs=_n_jobs
            )


    # pizza
    #     # must be ss csc
    #     # as of 24_12_16_07_28_00 not converting to csc.
    # @pytest.mark.parametrize('bad_X',
    #     (None, 'pd_df', 'ndarray', 'ss_csr', 'ss_coo')
    # )
    # def test_X_rejects_bad(
    #     self, _good_COLUMN, _X_ss, bad_X, _good_POLY_CSC, _good_equal_nan,
    #     _rtol_atol, _n_jobs
    # ):
    #

    #
    #     if bad_X is None:
    #         pass
    #     elif bad_X == 'pd_df':
    #         bad_X = pd.DataFrame(data=_X_ss.toarray())
    #     elif bad_X == 'ndarray':
    #         bad_X = _X_ss.toarray()
    #     elif bad_X == 'ss_csr':
    #         bad_X = _X_ss.tocsr()
    #     elif bad_X == 'ss_coo':
    #         bad_X = _X_ss.tocoo()
    #     else:
    #         raise Exception
    #
    #
    #     with pytest.raises(AssertionError):
    #         _get_dupls_for_combo_in_X_and_poly(
    #             _good_COLUMN,
    #             bad_X,
    #             _good_POLY_CSC,
    #             _good_equal_nan,
    #             *_rtol_atol,
    #             _n_jobs=_n_jobs
    #         )


    # END _X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # _POLY_CSC ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @pytest.mark.parametrize('junk_POLY_CSC',
        (-np.e, -1, 0, 1, np.e, True, False, 'trash', [0,1], (0,1), lambda x: x)
    )
    def test_POLY_CSC_rejects_junk(
        self, _good_COLUMN, _X_ss, junk_POLY_CSC, _good_equal_nan, _rtol_atol, _n_jobs
    ):

        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                _good_COLUMN,
                _X_ss,
                junk_POLY_CSC,
                _good_equal_nan,
                *_rtol_atol,
                _n_jobs=_n_jobs
            )


    @pytest.mark.parametrize('bad_POLY_CSC',
        (None, 'pd_df', 'ndarray', 'ss_csr', 'ss_coo')
    )
    def test_POLY_CSC_rejects_bad(
        self, _good_COLUMN, _X_ss, bad_POLY_CSC, _good_equal_nan, _rtol_atol, _n_jobs
    ):

        # must be ss csc

        if bad_POLY_CSC is None:
            pass
        elif bad_POLY_CSC == 'pd_df':
            bad_POLY_CSC = pd.DataFrame(data=_X_ss.toarray())
        elif bad_POLY_CSC == 'ndarray':
            bad_POLY_CSC = _X_ss.toarray()
        elif bad_POLY_CSC == 'ss_csr':
            bad_POLY_CSC = _X_ss.tocsr()
        elif bad_POLY_CSC == 'ss_coo':
            bad_POLY_CSC = _X_ss.tocoo()
        else:
            raise Exception

        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                _good_COLUMN,
                _X_ss,
                bad_POLY_CSC,
                _good_equal_nan,
                *_rtol_atol,
                _n_jobs=_n_jobs
            )

    # END _POLY_CSC ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # equal_nan  ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @pytest.mark.parametrize('junk_equal_nan',
        (-1,0,1,3.14,None,min,'trash',[0,1],{0,1}, (1,), {'a':1}, lambda x: x)
    )
    def test_rejects_junk(
        self, junk_equal_nan, _good_COLUMN, _X_ss, _good_POLY_CSC, _rtol_atol, _n_jobs
    ):

        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                _good_COLUMN,
                _X_ss,
                _good_POLY_CSC,
                junk_equal_nan,
                *_rtol_atol,
                _n_jobs=_n_jobs
            )

    # END equal_nan  ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    # rtol atol ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    @pytest.mark.parametrize('junk_rtol',
        (None, 'trash', [0,1], (0,1), {0,1}, {'a':1}, lambda x: x)
    )
    def test_rtol_rejects_junk(
        self, _good_COLUMN, _X_ss, _good_POLY_CSC, _good_equal_nan, junk_rtol, _n_jobs
    ):

        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                _good_COLUMN,
                _X_ss,
                _good_POLY_CSC,
                _good_equal_nan,
                _rtol=junk_rtol,
                _atol=1e-8,
                _n_jobs=_n_jobs
            )


    @pytest.mark.parametrize('junk_atol',
        (None, 'trash', [0, 1], (0, 1), {0, 1}, {'a': 1}, lambda x: x)
    )
    def test_atol_rejects_junk(
        self, _good_COLUMN, _X_ss, _good_POLY_CSC, _good_equal_nan, junk_atol, _n_jobs
    ):

        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                _good_COLUMN,
                _X_ss,
                _good_POLY_CSC,
                _good_equal_nan,
                _rtol=1e-5,
                _atol=junk_atol,
                _n_jobs=_n_jobs
            )


    @pytest.mark.parametrize('bad_rtol', (-np.e, -1, True, False))
    def test_rtol_rejects_bad(
        self, _good_COLUMN, _X_ss, _good_POLY_CSC, _good_equal_nan, bad_rtol, _n_jobs
    ):

        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                _good_COLUMN,
                _X_ss,
                _good_POLY_CSC,
                _good_equal_nan,
                _rtol=bad_rtol,
                _atol=1e-8,
                _n_jobs=_n_jobs
            )


    @pytest.mark.parametrize('bad_atol',
        (None, 'trash', [0, 1], (0, 1), {0, 1}, {'a': 1}, lambda x: x)
    )
    def test_atol_rejects_bad(
        self, _good_COLUMN, _X_ss, _good_POLY_CSC, _good_equal_nan, bad_atol, _n_jobs
    ):

        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                _good_COLUMN,
                _X_ss,
                _good_POLY_CSC,
                _good_equal_nan,
                _rtol=1e-5,
                _atol=bad_atol,
                _n_jobs=_n_jobs
            )





    # END rtol atol ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    # n_jobs ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('junk_n_jobs',
        (True, False, 'trash', [1, 2], {1, 2}, {'a': 1}, lambda x: x, min)
    )
    def test_junk_n_jobs(
        self, _good_COLUMN, _X_ss, _good_POLY_CSC, _good_equal_nan, _rtol_atol, junk_n_jobs
    ):

        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                _good_COLUMN,
                _X_ss,
                _good_POLY_CSC,
                _good_equal_nan,
                *_rtol_atol,
                _n_jobs=junk_n_jobs
            )

    @pytest.mark.parametrize('bad_n_jobs', [-2, 0])
    def test_bad_n_jobs(
        self, _good_COLUMN, _X_ss, _good_POLY_CSC, _good_equal_nan, _rtol_atol, bad_n_jobs
    ):

        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                _good_COLUMN,
                _X_ss,
                _good_POLY_CSC,
                _good_equal_nan,
                *_rtol_atol,
                _n_jobs=bad_n_jobs
            )

    # END n_jobs ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @pytest.mark.parametrize('_good_X',
        ('pd_df', 'ndarray', 'ss_csr', 'ss_csc')
    )
    @pytest.mark.parametrize(f'_equal_nan', (True, False))
    @pytest.mark.parametrize(f'_n_jobs', (-1, 1))
    def test_accepts_all_good(
        self, _good_COLUMN, _good_POLY_CSC, _X_ss, _good_X, _equal_nan, _rtol_atol, _n_jobs
    ):

        # pizza as of 24_12_16, now allowing np, pd, ss throughout partial_fit,
        # instead of converting to csc, to avoid copy / mutate of X
        # which means this must now accept all of them not just csc

        # _columns_getter now only allows ss that are indexable, dont test with
        # coo, dia, bsr

        if _good_X == 'pd_df':
            _good_X = pd.DataFrame(data=_X_ss.toarray())
        elif _good_X == 'ndarray':
            _good_X = _X_ss.toarray()
        elif _good_X == 'ss_csr':
            _good_X = _X_ss.tocsr()
        elif _good_X == 'ss_csc':
            _good_X = _X_ss.tocsc()
        else:
            raise Exception

        out = _get_dupls_for_combo_in_X_and_poly(
            _good_COLUMN,
            _good_X,
            _good_POLY_CSC,
            _equal_nan,
            *_rtol_atol,
            _n_jobs=_n_jobs
        )

        assert isinstance(out, list)
        assert len(out) == (_X_ss.shape[1] + _good_POLY_CSC.shape[1])
        assert all(map(isinstance, out, (bool for _ in out)))


class TestGetDuplsForComboAccuracy(Fixtures):


    def test_accuracy_no_dupls(
        self, _good_COLUMN, _X_ss, _good_POLY_CSC, _rtol_atol
    ):
        # with the fixtures, there should be no dupls in X or dupls

        out = _get_dupls_for_combo_in_X_and_poly(
            _good_COLUMN,
            _X_ss,
            _good_POLY_CSC,
            _equal_nan=True,
            _rtol=_rtol_atol[0],
            _atol=_rtol_atol[1],
            _n_jobs=1
        )

        assert isinstance(out, list)
        assert len(out) == (_X_ss.shape[1] + _good_POLY_CSC.shape[1])
        assert all(map(isinstance, out, (bool for _ in out)))
        assert sum(out) == 0



    def test_accuracy_dupls_in_X(self, _X_ss, _rtol_atol, _shape, _combos):

        # rig X to have some dupls to see if this finds them

        # simulate running the last combo (POLY IS ALMOST FULLY BUILT)

        _X_np = _X_ss.toarray()

        _X_np[:, 1] = _X_np[:, 0]
        _X_np[:, _shape[1] - 1] = _X_np[:, 0]
        # duplicates in the 0, 1, and -1 positions of X

        # for _COLUMN, use the last term in combos
        # must be ndarray!
        _POLY_COLUMN = _X_np[:, _combos[-1]].prod(1).ravel()

        _X = ss.csc_array(_X_np)

        # build CSC_ARRAY with all the other combos
        # cant have any duplicates in this
        _POLY = np.empty((_shape[0], 0))
        for _combo in _combos[:-1]:
            _COLUMN = _X_np[:, _combo].prod(1).reshape((-1, 1))
            for _poly_idx in range(_POLY.shape[1]):
                if np.array_equal(_COLUMN.ravel(), _POLY[:, _poly_idx]):
                    break
            else:
                _POLY = np.hstack((_POLY, _COLUMN))

        _POLY = ss.csc_array(_POLY)


        out = _get_dupls_for_combo_in_X_and_poly(
            _POLY_COLUMN,
            _X,
            _POLY,
            _equal_nan=True,
            _rtol=_rtol_atol[0],
            _atol=_rtol_atol[1],
            _n_jobs=1
        )

        assert isinstance(out, list)
        assert len(out) == (_X_ss.shape[1] + _POLY.shape[1])
        assert all(map(isinstance, out, (bool for _ in out)))
        assert sum(out) == 1


    def test_accuracy_dupls_in_poly(self, _rtol_atol):

        # rig a dataset to cause poly terms to be dupls to see if this finds them
        # this is conveniently done by doing poly on a dummied series and
        # only using interaction terms!

        # simulate running the last combo (POLY IS ALMOST FULLY BUILT)

        _shape = (100, 6)

        _pool = list('abcdefghijklmnopqrstuv')[:_shape[1]]

        while True:
            # ensure that X_np will dummy out to _shape[1] columns
            # that is, there are _shape[1] unique categories in _X_np
            _X_np = np.random.choice(_pool, _shape[0], replace=True).reshape((-1, 1))
            if len(np.unique(_X_np)) == _shape[1]:
                break

        _X_np = OneHotEncoder(drop=None, sparse_output=False).fit_transform(_X_np)

        assert _X_np.shape == _shape

        _combos = list(itertools.combinations(range(len(_pool)), 2))

        del _pool

        # for _COLUMN, use the last term in combos
        # must be ndarray!
        _POLY_COLUMN = _X_np[:, _combos[-1]].prod(1).ravel()

        _X_ss = ss.csc_array(_X_np)

        assert _X_ss.shape == _shape

        # build CSC_ARRAY with all the other combos
        # cant have any duplicates in this
        _POLY = np.empty((_shape[0], 0))
        for _combo in _combos[:-1]:
            _COLUMN = _X_np[:, _combo].prod(1).reshape((-1, 1))
            for _poly_idx in range(_POLY.shape[1]):
                if np.array_equal(_COLUMN.ravel(), _POLY[:, _poly_idx]):
                    break
            else:
                _POLY = np.hstack((_POLY, _COLUMN))


        # every poly combo should generate a vector of zeros.
        # there should be no duplicates against X. but every poly term
        # is a duplicate of each other, so _POLY should be one vector
        # of zeros.
        assert _POLY.shape[1] == 1

        assert np.array_equal(
            _POLY,
            np.zeros((_shape[0], 1)).reshape((-1,1))
        )

        _POLY = ss.csc_array(_POLY)


        out = _get_dupls_for_combo_in_X_and_poly(
            _POLY_COLUMN,
            _X_ss,
            _POLY,
            _equal_nan=True,
            _rtol=_rtol_atol[0],
            _atol=_rtol_atol[1],
            _n_jobs=1
        )

        assert isinstance(out, list)
        assert len(out) == (_X_ss.shape[1] + _POLY.shape[1])
        assert all(map(isinstance, out, (bool for _ in out)))
        assert sum(out[:_shape[1]]) == 0  # for the X part no duplicates
        # all values (only 1) for the poly part of 'out' should be True
        assert sum(out[_shape[1]:]) == _POLY.shape[1]









