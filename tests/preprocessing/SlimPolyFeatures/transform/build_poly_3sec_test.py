# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import itertools

import numpy as np
import scipy.sparse as ss

from pybear.preprocessing.SlimPolyFeatures._transform._build_poly \
    import _build_poly

import pytest





class TestBuildPoly:


    # def _build_poly(
    #     X: DataType,
    #     _active_combos: tuple[tuple[int, ...], ...],
    #     _n_jobs: Union[numbers.Integral, None]
    # ) -> ss.csc_array:


    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (10,5)


    @staticmethod
    @pytest.fixture(scope='module')
    def _good_active_combos(_shape):
        # draw must be >= 2, SPF doesnt allow _degree < 2 in poly
        return tuple(itertools.combinations_with_replacement(range(_shape[1]), 2))


    @staticmethod
    @pytest.fixture(scope='module')
    def _good_X(_shape):
        return ss.csc_array(np.random.randint(0, 3, _shape))


    @pytest.mark.parametrize('junk_inputs',
        (-2.7, -1, 0, 1, 2.7, True, False, None, 'junk', [0,1], (0,1),
         {0,1}, {'a': 1}, lambda x: x)
    )
    def test_minimalist_validation(
        self, junk_inputs, _good_active_combos, _good_X, _shape
    ):

        # X
        with pytest.raises(AssertionError):
            _build_poly(
                X=junk_inputs,
                _active_combos=_good_active_combos
            )

        # active_combos
        with pytest.raises(AssertionError):
            _build_poly(
                X=_good_X,
                _active_combos=junk_inputs
            )

        # END validation - - - - - - - - - - - - - - - - - - - - - - - - - - - -


    @pytest.mark.parametrize('_n_jobs', (-1, 1, 2, 3, 4, None))
    def test_accepts_good_params(self, _good_X, _good_active_combos, _n_jobs):

        _build_poly(
            X=_good_X,
            _active_combos=_good_active_combos
        )


    def test_build_poly(self, _good_X, _good_active_combos, _shape):

        out =  _build_poly(
            _good_X,
            _good_active_combos
        )

        assert isinstance(out, ss.csc_array)
        assert out.shape == (_shape[0], len(_good_active_combos))


        # build a referee output - - - - - - - - - - - -
        ref = np.empty((_shape[0], 0))

        _column_pool = _good_X.toarray()

        for _combo in _good_active_combos:
            ref = np.hstack((
                ref, _column_pool[:, _combo].prod(1).reshape((-1,1))
            ))
        # END build a referee output - - - - - - - - - - - -

        out = out.toarray()

        assert out.shape == ref.shape

        assert np.array_equal(out, ref)














