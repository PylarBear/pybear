# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.SlimPolyFeatures._partial_fit.\
    _get_dupls_for_combo_in_X_and_poly import _get_dupls_for_combo_in_X_and_poly


import numpy as np

import pytest


pytest.skip(reason=f"pizza isnt done!", allow_module_level=True)



class Fixtures:


    # fixtures ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (10, 20)


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
        # otherwise is as passed (np or pd)

        def foo(_format, _dtype, _has_nan, _dupls):

            return _X_factory(
                _dupl=_dupls,
                _has_nan=_has_nan,
                _format=_format,
                _dtype=_dtype,
                _shape=_shape
            )


        return foo


    @staticmethod
    @pytest.fixture(scope='module')
    def _good_COLUMN(_X_ss):
        return _X_ss[:, [0,1,2]].prod(1)


    @staticmethod
    @pytest.fixture(scope='module')
    def _good_POLY_CSC(_X_ss):
        return _X_ss[:, [0,1,2]]


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
    @pytest.mark.parametrize('_COLUMN',
        (-np.e, -1, 0, 1, np.e, True, False, 'trash', [0,1], (0,1), lambda x: x)
    )
    def test_COLUMN_rejects_junk(self, _COLUMN, _X_ss, _good_POLY_CSC, _rtol_atol, _n_jobs):
        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                _COLUMN,
                _X=_X_ss,
                _POLY_CSC=_good_POLY_CSC,
                _equal_nan=True,
                _rtol=_rtol_atol[0],
                _atol=_rtol_atol[1],
                _n_jobs=_n_jobs
            )


    @pytest.mark.parametrize('_COLUMN',
        ({'a':0, 'b': 1, 'c':np.nan}, {0: 1, 1:0, 2: np.nan})
    )
    def test_COLUMN_rejects_bad(self, _COLUMN, _rtol_atol, _n_jobs):
        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                _COLUMN,
                _X,
                _POLY_CSC,
                _equal_nan,
                *_rtol_atol,
                _n_jobs=_n_jobs
            )


    @pytest.mark.parametrize('_COLUMN',
        (None, {}, {(0,): 1, (4,): 0, (5,): 0}, {(0,): 1, (2,): np.nan})
    )
    def test_COLUMN_accepts_good(self, _COLUMN, _X_ss, _rtol_atol, _n_jobs):
        _get_dupls_for_combo_in_X_and_poly(
            _COLUMN,
            _X,
            _POLY_CSC,
            _equal_nan,
            *_rtol_atol,
            _n_jobs=_n_jobs
        )
    # END _COLUMN ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # _X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @pytest.mark.parametrize('_X',
        (-np.e, -1, 0, 1, np.e, True, False, 'trash', [0,1], (0,1), lambda x: x)
    )
    def test_X_rejects_junk(self, _X, _rtol_atol, _n_jobs):
        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                _COLUMN,
                _X,
                _POLY_CSC,
                _equal_nan,
                *_rtol_atol,
                _n_jobs=_n_jobs
            )


    @pytest.mark.parametrize('_X',
        (None, {'a':0, 'b': 1, 'c':np.nan}, {0: 1, 1:0, 2: np.nan})
    )
    def test_X_rejects_bad(self, _X, _rtol_atol, _n_jobs):
        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                _COLUMN,
                _X,
                _POLY_CSC,
                _equal_nan,
                *_rtol_atol,
                _n_jobs=_n_jobs
            )


    @pytest.mark.parametrize('_X',
        ({}, {(0,): 1, (4,): 0, (5,): 0}, {(0,7): 1, (2,8): np.nan})
    )
    def test_X_accepts_good(self, _X, _rtol_atol, _n_jobs):
        _get_dupls_for_combo_in_X_and_poly(
            _COLUMN,
            _X,
            _POLY_CSC,
            _equal_nan,
            *_rtol_atol,
            _n_jobs=_n_jobs
        )

    # END _X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # _POLY_CSC ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @pytest.mark.parametrize('_POLY_CSC',
        (-np.e, -1, 0, 1, np.e, True, False, 'trash', [0,1], (0,1), lambda x: x)
    )
    def test_POLY_CSC_rejects_junk(self, _POLY_CSC, _rtol_atol, _n_jobs):
        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                _COLUMN,
                _X,
                _POLY_CSC,
                _equal_nan,
                *_rtol_atol,
                _n_jobs=_n_jobs
            )


    @pytest.mark.parametrize('_POLY_CSC',
        (None, {'a':0, 'b': 1, 'c':np.nan}, {0: 1, 1:0, 2: np.nan})
    )
    def test_POLY_CSC_rejects_bad(self, _POLY_CSC, _rtol_atol, _n_jobs):
        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                _COLUMN,
                _X,
                _POLY_CSC,
                _equal_nan,
                *_rtol_atol,
                _n_jobs=_n_jobs
            )


    @pytest.mark.parametrize('_POLY_CSC',
        ({}, {(0,): 1, (4,): 0, (5,): 0}, {(0,7): 1, (2,8): np.nan})
    )
    def test_POLY_CSC_accepts_good(self, _POLY_CSC, _rtol_atol, _n_jobs):
        _get_dupls_for_combo_in_X_and_poly(
            _COLUMN,
            _X,
            _POLY_CSC,
            _equal_nan,
            *_rtol_atol,
            _n_jobs=_n_jobs
        )

    # END _POLY_CSC ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **



    # rtol atol ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    @pytest.mark.parametrize('_rtol',
        (None, 'trash', [0,1], (0,1), {0,1}, {'a':1}, lambda x: x)
    )
    def test_rtol_rejects_junk(self, _rtol, _n_jobs):

        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                _COLUMN,
                _X,
                _POLY_CSC,
                _equal_nan,
                _rtol=_rtol,
                _atol=1e-8,
                _n_jobs=_n_jobs
            )


    @pytest.mark.parametrize('_atol',
        (None, 'trash', [0, 1], (0, 1), {0, 1}, {'a': 1}, lambda x: x)
    )
    def test_atol_rejects_junk(self, _atol, _n_jobs):
        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                _COLUMN,
                _X,
                _POLY_CSC,
                _equal_nan,
                _rtol=1e-5,
                _atol=_atol,
                _n_jobs=_n_jobs
            )


    @pytest.mark.parametrize('_rtol', (-np.e, -1, True, False))
    def test_rtol_rejects_bad(self, _rtol, _n_jobs):
        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                _COLUMN,
                _X,
                _POLY_CSC,
                _equal_nan,
                _rtol=_rtol,
                _atol=1e-8,
                _n_jobs=_n_jobs
            )

    @pytest.mark.parametrize('_atol',
        (None, 'trash', [0, 1], (0, 1), {0, 1}, {'a': 1}, lambda x: x)
    )
    def test_atol_rejects_bad(self, _atol, _n_jobs):
        with pytest.raises(AssertionError):
            _get_dupls_for_combo_in_X_and_poly(
                _COLUMN,
                _X,
                _POLY_CSC,
                _equal_nan,
                _rtol=1e-5,
                _atol=_atol,
                _n_jobs=_n_jobs
            )


    @pytest.mark.parametrize('_rtol', (0, 1e-5, 1, np.e))
    def test_rtol_accepts_good(self, _rtol, _n_jobs):

        _get_dupls_for_combo_in_X_and_poly(
            _COLUMN,
            _X,
            _POLY_CSC,
            _equal_nan,
            _rtol=_rtol,
            _atol=1e-8,
            _n_jobs=_n_jobs
        )

    @pytest.mark.parametrize('_atol', (0, 1e-5, 1, np.e))
    def test_atol_accepts_good(self, _atol, _n_jobs):

        _get_dupls_for_combo_in_X_and_poly(
            _COLUMN,
            _X,
            _POLY_CSC,
            _equal_nan,
            _rtol=1e-5,
            _atol=_atol,
            _n_jobs=_n_jobs
        )

    # END rtol atol ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    # n_jobs ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('junk_n_jobs',
        (True, False, 'trash', [1, 2], {1, 2}, {'a': 1}, lambda x: x, min)
    )
    def test_junk_n_jobs(self, _X_ss, _rtol_atol, junk_n_jobs):

        with pytest.raises(TypeError):
            _get_dupls_for_combo_in_X_and_poly(
                _COLUMN,
                _X,
                _POLY_CSC,
                _equal_nan,
                *_rtol_atol,
                _n_jobs=_n_jobs
            )

    @pytest.mark.parametrize('bad_n_jobs', [-2, 0])
    def test_bad_n_jobs(self, _X_ss, _rtol_atol, bad_n_jobs):

        with pytest.raises(ValueError):
            _get_dupls_for_combo_in_X_and_poly(
                _COLUMN,
                _X,
                _POLY_CSC,
                _equal_nan,
                *_rtol_atol,
                _n_jobs=_n_jobs
            )


    @pytest.mark.parametrize('good_n_jobs', [-1, 1, 10, None])
    def test_good_n_jobs(self, _X_ss, _rtol_atol, good_n_jobs):

        _get_dupls_for_combo_in_X_and_poly(
            _COLUMN,
            _X,
            _POLY_CSC,
            _equal_nan,
            *_rtol_atol,
            _n_jobs=_n_jobs
        )

    # END n_jobs ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *



class TestGetDuplsForComboAccuracy(Fixtures):

    pass



















