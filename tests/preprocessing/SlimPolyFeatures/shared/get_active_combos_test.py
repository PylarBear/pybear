# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from pybear.preprocessing.SlimPolyFeatures._partial_fit. \
    _get_active_combos import _get_active_combos

import itertools
import numpy as np

import pytest



class TestGetActiveCombos:

    # all that happens is that combos that are in poly_constants_ or
    # dropped_poly_duplicates_ are omitted from combos

    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (100, 10)


    @staticmethod
    @pytest.fixture(scope='module')
    def _combo_getter(_shape):

        def foo(_min_degree, _degree, _interaction_only):

            _fxn = itertools.combinations if \
                _interaction_only else itertools.combinations_with_replacement

            out = []
            for _degree in range(_min_degree, _degree + 1):
                out += list(_fxn(range(_shape[1]), _degree))

            return out

        return foo


    @pytest.mark.parametrize('_min_degree, _degree',
        ((1, 2), (1,3), (2,3), (2,4))
    )
    @pytest.mark.parametrize('_interaction_only', (True, False))
    @pytest.mark.parametrize('_poly_constants',
        ('empty', 'trial_1', 'trial_2', 'trial_3')
    )
    @pytest.mark.parametrize('_dropped_poly_duplicates',
        ('empty', 'trial_1', 'trial_2', 'trial_3')
    )
    def test_accuracy(
        self, _combo_getter, _min_degree, _degree, _interaction_only,
        _poly_constants, _dropped_poly_duplicates
    ):

        _combos = _combo_getter(_min_degree, _degree, _interaction_only)

        # create some poly constants and dropped duplicates based
        # on arbitrary selection from _combos
        # there can be no repeated combos within or between constants and duplicates

        if _poly_constants == 'empty':
            _poly_constants = {}
        elif _poly_constants == 'trial_1':
            _poly_constants = {(0,1):1}
        elif _poly_constants == 'trial_2':
            _poly_constants = {(0,1):1, (0,2):2}
        elif _poly_constants == 'trial_3':
            _poly_constants = {(0,1):1, (1,1):1, (2,3):np.nan}
        else:
            raise Exception

        if _dropped_poly_duplicates == 'empty':
            _dropped_poly_duplicates = {}
        elif _dropped_poly_duplicates == 'trial_1':
            _dropped_poly_duplicates = {(6,7): (5,)}
        elif _dropped_poly_duplicates == 'trial_2':
            _dropped_poly_duplicates = {(8,9): (8,7)}
        elif _dropped_poly_duplicates == 'trial_3':
            _dropped_poly_duplicates = {(6,7): (5,), (6,8): (5,), (8,9): (8,7)}
        else:
            raise Exception

        out = _get_active_combos(
            _combos,
            _poly_constants,
            _dropped_poly_duplicates
        )

        for _combo_tuple in _poly_constants:
            assert _combo_tuple not in out

        for _combo_tuple in _dropped_poly_duplicates:
            assert _combo_tuple not in out
















