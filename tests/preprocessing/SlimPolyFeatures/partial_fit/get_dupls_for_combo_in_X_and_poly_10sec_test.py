# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import itertools

import numpy as np
import pandas as pd
import scipy.sparse as ss
import dask.array as da
import dask.dataframe as ddf

from sklearn.preprocessing import OneHotEncoder

from pybear.preprocessing._SlimPolyFeatures._partial_fit.\
    _get_dupls_for_combo_in_X_and_poly import _get_dupls_for_combo_in_X_and_poly



class Fixtures:


    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @staticmethod
    @pytest.fixture(scope='module')
    def _gdfc_args():

        _args = {}
        _args['_equal_nan'] = True
        _args['_rtol'] = 1e-5
        _args['_atol'] = 1e-8
        _args['_n_jobs'] = 1  # leave this at 1 because of contention

        return _args


    @staticmethod
    @pytest.fixture(scope='module')
    def _combos(_shape):
        return list(itertools.combinations_with_replacement(range(_shape[1]), 2))


    @staticmethod
    @pytest.fixture(scope='module')
    def _good_COLUMN(X_np, _combos):
        # must be ndarray!
        # make _COLUMN from the last combo in _combos
        # when building _good_CSC_ARRAY fixture, build it with all the combos
        # except the last one. So it's like were doing the last scan of a
        # partial fit with the last combo across all the other polys.
        return X_np[:, _combos[-1]].prod(1)


    @staticmethod
    @pytest.fixture(scope='module')
    def _good_POLY_CSC(X_np, _shape, _combos):

        # cant have any duplicates in this
        # make _COLUMN from the last combo in _combos
        # when building _good_POLY_CSC fixture, build it with all the combos
        # except the last one.

        _POLY = np.empty((_shape[0], 0))
        for _combo in _combos[:-1]:
            _COLUMN =  X_np[:, _combo].prod(1).reshape((-1,1))
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
    #     _X: InternalDataContainer,
    #     _POLY_CSC: Union[ss.csc_array, ss.csc_matrix],
    #     _min_degree: pizza,
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
        self, junk_COLUMN, X_np, _good_POLY_CSC, _gdfc_args
    ):

        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                junk_COLUMN,
                _X=X_np,
                _POLY_CSC=_good_POLY_CSC,
                _min_degree=1,  # pizza
                **_gdfc_args
            )


    @pytest.mark.parametrize('bad_COLUMN', ('pd_series', 'ss_csr', 'ss_coo'))
    def test_COLUMN_rejects_bad(self, bad_COLUMN, X_np, _good_POLY_CSC, _gdfc_args):

        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                bad_COLUMN,
                X_np,
                1,  # pizza _min_degree
                _good_POLY_CSC,
                **_gdfc_args
            )


    # END _COLUMN ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # _X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @pytest.mark.parametrize('junk_X',
        (-np.e, -1, 0, 1, np.e, True, False, 'trash', [0,1], (0,1), lambda x: x)
    )
    def test_X_rejects_junk(self, _good_COLUMN, junk_X, _good_POLY_CSC, _gdfc_args):

        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                _good_COLUMN,
                junk_X,
                _good_POLY_CSC,
                1, # pizza _min_degree
                **_gdfc_args
            )


    @pytest.mark.parametrize('bad_X', (None, 'dask_df', 'dask_array'))
    def test_X_rejects_bad(
        self, _good_COLUMN, X_np, bad_X, _good_POLY_CSC, _gdfc_args, _shape
    ):

        if bad_X is None:
            pass
        elif bad_X == 'dask_df':
            bad_X = ddf.from_array(X_np).repartition((_shape[0]//2, ))
        elif bad_X == 'dask_array':
            bad_X = da.from_array(X_np).rechunk((_shape[0]//2, _shape[1]))
        else:
            raise Exception


        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                _good_COLUMN,
                bad_X,
                _good_POLY_CSC,
                1,   # pizza _min_degree
                **_gdfc_args
            )


    @pytest.mark.parametrize('_format',
        ('coo_matrix', 'coo_array', 'dia_matrix',
         'dia_array', 'bsr_matrix', 'bsr_array')
    )
    def test_X_rejects_coo_dia_bsr(
        self, _X_factory, _shape, _format, _good_COLUMN, _good_POLY_CSC,
        _gdfc_args
    ):

        _bad_X = _X_factory(
            _dupl=None,
            _has_nan=False,
            _format=_format,
            _dtype='flt',
            _columns=None,
            _constants=None,
            _noise=0,
            _zeros=None,
            _shape=_shape
        )

        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                _good_COLUMN,
                _bad_X,
                _good_POLY_CSC,
                1,   # pizza _min_degree
                **_gdfc_args
            )

    # END _X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # _POLY_CSC ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @pytest.mark.parametrize('junk_POLY_CSC',
        (-np.e, -1, 0, 1, np.e, True, False, 'trash', [0,1], (0,1), lambda x: x)
    )
    def test_POLY_CSC_rejects_junk(
        self, _good_COLUMN, X_np, junk_POLY_CSC, _gdfc_args
    ):

        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                _good_COLUMN,
                X_np,
                junk_POLY_CSC,
                1,   # pizza _min_degree
                **_gdfc_args
            )


    @pytest.mark.parametrize('bad_POLY_CSC',
        (None, 'pd_df', 'ndarray', 'ss_csr', 'ss_coo')
    )
    def test_POLY_CSC_rejects_bad(
        self, _good_COLUMN, X_np, bad_POLY_CSC, _gdfc_args
    ):

        # must be ss csc

        if bad_POLY_CSC is None:
            pass
        elif bad_POLY_CSC == 'pd_df':
            bad_POLY_CSC = pd.DataFrame(data=X_np)
        elif bad_POLY_CSC == 'ndarray':
            bad_POLY_CSC = X_np
        elif bad_POLY_CSC == 'ss_csr':
            bad_POLY_CSC = ss.csr_array(X_np)
        elif bad_POLY_CSC == 'ss_coo':
            bad_POLY_CSC = ss.coo_array(X_np)
        else:
            raise Exception

        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                _good_COLUMN,
                X_np,
                bad_POLY_CSC,
                1,   # pizza _min_degree
                **_gdfc_args
            )

    # END _POLY_CSC ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # equal_nan  ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @pytest.mark.parametrize('junk_equal_nan',
        (-1,0,1,3.14,None,min,'trash',[0,1],{0,1}, (1,), {'a':1}, lambda x: x)
    )
    def test_rejects_junk(
        self, junk_equal_nan, _good_COLUMN, X_np, _good_POLY_CSC, _gdfc_args
    ):

        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                _good_COLUMN,
                X_np,
                _good_POLY_CSC,
                1,  # pizza _min_degree
                junk_equal_nan,
                _gdfc_args['_rtol'],
                _gdfc_args['_atol'],
                _gdfc_args['_n_jobs']
            )

    # END equal_nan  ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    # rtol atol ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    @pytest.mark.parametrize('junk_rtol',
        (None, 'trash', [0,1], (0,1), {0,1}, {'a':1}, lambda x: x)
    )
    def test_rtol_rejects_junk(
        self, _good_COLUMN, X_np, _good_POLY_CSC, junk_rtol, _gdfc_args
    ):

        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                _good_COLUMN,
                X_np,
                _good_POLY_CSC,
                1, # pizza _min_degree
                _gdfc_args['_equal_nan'],
                junk_rtol,
                _gdfc_args['_atol'],
                _gdfc_args['_n_jobs']
            )


    @pytest.mark.parametrize('junk_atol',
        (None, 'trash', [0, 1], (0, 1), {0, 1}, {'a': 1}, lambda x: x)
    )
    def test_atol_rejects_junk(
        self, _good_COLUMN, X_np, _good_POLY_CSC, junk_atol, _gdfc_args
    ):

        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                _good_COLUMN,
                X_np,
                _good_POLY_CSC,
                1,   # pizza _min_degree
                _gdfc_args['_equal_nan'],
                _gdfc_args['_rtol'],
                junk_atol,
                _gdfc_args['_n_jobs']
            )


    @pytest.mark.parametrize('bad_rtol', (-np.e, -1, True, False))
    def test_rtol_rejects_bad(
        self, _good_COLUMN, X_np, _good_POLY_CSC, bad_rtol, _gdfc_args
    ):

        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                _good_COLUMN,
                X_np,
                _good_POLY_CSC,
                1,  # pizza _min_degree
                _gdfc_args['_equal_nan'],
                bad_rtol,
                _gdfc_args['_atol'],
                _gdfc_args['_n_jobs']
            )


    @pytest.mark.parametrize('bad_atol',
        (None, 'trash', [0, 1], (0, 1), {0, 1}, {'a': 1}, lambda x: x)
    )
    def test_atol_rejects_bad(
        self, _good_COLUMN, X_np, _good_POLY_CSC, bad_atol, _gdfc_args
    ):

        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                _good_COLUMN,
                X_np,
                _good_POLY_CSC,
                1,  # pizza _min_degree
                _gdfc_args['_equal_nan'],
                _gdfc_args['_rtol'],
                bad_atol,
                _gdfc_args['_n_jobs']
            )

    # END rtol atol ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    # n_jobs ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('junk_n_jobs',
        (True, False, 'trash', [1, 2], {1, 2}, {'a': 1}, lambda x: x, min)
    )
    def test_junk_n_jobs(
        self, _good_COLUMN, X_np, _good_POLY_CSC, _gdfc_args, junk_n_jobs
    ):

        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                _good_COLUMN,
                X_np,
                _good_POLY_CSC,
                1,  # pizza _min_degree
                _gdfc_args['_equal_nan'],
                _gdfc_args['_rtol'],
                _gdfc_args['_atol'],
                junk_n_jobs
            )

    @pytest.mark.parametrize('bad_n_jobs', [-2, 0])
    def test_bad_n_jobs(
        self, _good_COLUMN, X_np, _good_POLY_CSC, _gdfc_args, bad_n_jobs
    ):

        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                _good_COLUMN,
                X_np,
                _good_POLY_CSC,
                1,  # pizza _min_degree
                _gdfc_args['_equal_nan'],
                _gdfc_args['_rtol'],
                _gdfc_args['_atol'],
                bad_n_jobs
            )

    # END n_jobs ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    @pytest.mark.parametrize('_good_X',
        ('pd_df', 'ndarray', 'ss_csr', 'ss_csc')
    )
    @pytest.mark.parametrize(f'_equal_nan', (True, False))
    @pytest.mark.parametrize(f'_n_jobs', (-1, 1))
    def test_accepts_all_good(
        self, _good_COLUMN, _good_POLY_CSC, X_np, _good_X, _equal_nan,
        _gdfc_args, _n_jobs
    ):

        # X in original form as np, pd, ss is carried throughout partial_fit,
        # instead of converting to csc, to avoid copy / mutate of X
        # which means this must now accept all of them not just csc

        # _columns_getter now only allows ss that are indexable, dont test with
        # coo, dia, bsr

        if _good_X == 'pd_df':
            _good_X = pd.DataFrame(data=X_np)
        elif _good_X == 'ndarray':
            _good_X = X_np
        elif _good_X == 'ss_csr':
            _good_X = ss.csr_array(X_np)
        elif _good_X == 'ss_csc':
            _good_X = ss.csc_array(X_np)
        else:
            raise Exception

        out = _get_dupls_for_combo_in_X_and_poly(
            _good_COLUMN,
            _good_X,
            _good_POLY_CSC,
            1,  # pizza _min_degree
            _equal_nan,
            _gdfc_args['_rtol'],
            _gdfc_args['_atol'],
            _n_jobs
        )

        assert isinstance(out, list)
        assert len(out) == (X_np.shape[1] + _good_POLY_CSC.shape[1])
        assert all(map(isinstance, out, (bool for _ in out)))


class TestGetDuplsForComboAccuracy(Fixtures):


    def test_accuracy_no_dupls(
        self, _good_COLUMN, X_np, _good_POLY_CSC, _gdfc_args
    ):
        # with the fixtures, there should be no dupls in X or dupls

        out = _get_dupls_for_combo_in_X_and_poly(
            _good_COLUMN,
            X_np,
            _good_POLY_CSC,
            1,   # pizza _min_degree
            **_gdfc_args
        )

        assert isinstance(out, list)
        assert len(out) == (X_np.shape[1] + _good_POLY_CSC.shape[1])
        assert all(map(isinstance, out, (bool for _ in out)))
        assert sum(out) == 0


    def test_accuracy_dupls_in_X(
        self, X_np, _shape, _combos, _good_COLUMN, _good_POLY_CSC, _gdfc_args
    ):

        # rig X to have some dupls to see if this finds them

        # simulate running the last combo (POLY IS ALMOST FULLY BUILT)

        X_np[:, 0] = _good_COLUMN
        X_np[:, 1] = _good_COLUMN
        X_np[:, _shape[1] - 1] = _good_COLUMN
        # duplicates of _COLUMN in the 0, 1, and -1 positions of X

        out = _get_dupls_for_combo_in_X_and_poly(
            _good_COLUMN,
            X_np,
            _good_POLY_CSC,  # this doesnt actually match against modified X_np
            1,  # pizza _min_degree
            **_gdfc_args
        )

        # in this contrived arrangement, _COLUMN should have 3 hits against X
        # and zero against _POLY

        assert isinstance(out, list)
        assert len(out) == (X_np.shape[1] + _good_POLY_CSC.shape[1])
        assert all(map(isinstance, out, (bool for _ in out)))
        assert sum(out) == 3
        assert out[0] is True
        assert out[1] is True
        assert out[_shape[1]-1] is True
        assert sum(out[_shape[1]:]) == 0


    def test_accuracy_dupls_in_poly(self, _gdfc_args):

        # rig a dataset to cause poly terms to be dupls.
        # this is conveniently done by doing poly on a dummied
        # series and only using interaction terms

        # THIS IS TESTING USING A FICTITIOUS STATE OF POLY!
        # we are going to intentionally build POLY with duplicates in it
        # (which should not happen in SPF) and those duplicates will also
        # happen to be columns of constants (which SPF should never allow
        # into POLY!)

        # This is just to see if the POLY scan part of _get_dupls can
        # actually find duplicates.

        # simulate running the last combo (POLY IS ALMOST FULLY BUILT)

        _shape = (100, 6)

        _pool = list('abcdefghijklmnopqrstuv')[:_shape[1]]

        while True:
            # ensure that X_np will dummy out to _shape[1] columns
            # that is, there are _shape[1] unique categories in _X_np
            _X_np = np.random.choice(_pool, _shape[0], replace=True)
            if len(np.unique(_X_np)) == _shape[1]:
                break

        _X_np = _X_np.reshape((-1, 1))

        _X_np = OneHotEncoder(drop=None, sparse_output=False).fit_transform(_X_np)

        assert _X_np.shape == _shape

        _combos = list(itertools.combinations(range(len(_pool)), 2))

        del _pool

        # for _COLUMN, use the last term in combos
        # must be ndarray!
        _POLY_COLUMN = _X_np[:, _combos[-1]].prod(1).ravel()

        # build CSC_ARRAY with all the other preceding combos
        # BUT LEAVE THE DUPLICATES IN THIS!
        _POLY = np.empty((_shape[0], 0))
        for _combo in _combos[:-1]:
            _POLY = np.hstack((
                _POLY,
                _X_np[:, _combo].prod(1).reshape((-1, 1))
            ))


        # every poly combo should have generated a vector of zeros, which
        # means _POLY should be full of columns of zeros.

        assert _POLY.shape[1] == len(_combos) - 1  # last combo not in POLY yet!

        assert np.array_equal(
            _POLY,
            np.zeros((_shape[0], len(_combos) - 1))
        )

        _POLY = ss.csc_array(_POLY)

        # there should be no duplicates against X. but every poly term
        # is a duplicate of each other and of _COLUMN, so this fictitious
        # _POLY should be marked as duplicate for all positions (_COLUMN
        # is a duplicate of all positions).
        out = _get_dupls_for_combo_in_X_and_poly(
            _POLY_COLUMN,
            _X_np,
            _POLY,
            1,  # pizza _min_degree
            **_gdfc_args
        )

        assert isinstance(out, list)
        assert len(out) == (_X_np.shape[1] + _POLY.shape[1])
        assert all(map(isinstance, out, (bool for _ in out)))
        assert sum(out[:_shape[1]]) == 0  # for the X part no duplicates

        assert len(out[_shape[1]:]) == len(_combos) - 1 # last combo not in POLY

        # all values for the poly part of 'out' should be True
        assert sum(out[_shape[1]:]) == len(_combos) - 1 # last combo not in POLY





