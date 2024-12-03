# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from pybear.preprocessing.NoDupPolyFeatures._base_fit._merge_combo_dupls import (
    _merge_combo_dupls
)


import pytest
import numpy as np


pytest.skip(reason=f"pizza is half baked!", allow_module_level=True)




class TestMergeComboDupls:


    @staticmethod
    @pytest.fixture(scope='module')
    def _good_combo_dupls():
        return [(1,), (1,2)]


    @staticmethod
    @pytest.fixture(scope='module')
    def _good_partialfit_dupls():
        return [[(1, ), (2, )]]



    # _dupls_for_this_combo ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    @pytest.mark.parametrize('_dupls_for_this_combo',
        (-np.e, -1, 0, 1, np.e, None, True, False, 'trash', lambda x: x)
    )
    def test_dupls_for_this_combo_rejects_junk(self, _dupls_for_this_combo, _good_partialfit_dupls):
        with pytest.raises(AssertionError):
            _merge_combo_dupls(
                _dupls_for_this_combo,
                _good_partialfit_dupls
            )


    @pytest.mark.parametrize('_dupls_for_this_combo',
        ((), ((0,), (1,)), {(0,), (1,)}, [[0], [1]])
    )
    def test_dupls_for_this_combo_rejects_bad(self, _dupls_for_this_combo, _good_partialfit_dupls):
        with pytest.raises(AssertionError):
            _merge_combo_dupls(
                _dupls_for_this_combo,
                _good_partialfit_dupls
            )


    @pytest.mark.parametrize('_dupls_for_this_combo',
        ([], [(0,)], [(0,), (1,)], [(2,), (0,1,2)])
    )
    def test_dupls_for_this_combo_accepts_good(self, _dupls_for_this_combo, _good_partialfit_dupls):
        _merge_combo_dupls(
            _dupls_for_this_combo,
            _good_partialfit_dupls
        )
    # END _dupls_for_this_combo ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # _poly_dupls_current_partial_fit ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    @pytest.mark.parametrize('_poly_dupls_current_partial_fit',
        (-np.e, -1, 0, 1, np.e, None, True, False, 'trash', lambda x: x)
    )
    def test_poly_dupls_current_partial_fit_rejects_junk(
        self, _good_dupls_for_this_combo, _poly_dupls_current_partial_fit
    ):

        with pytest.raises(AssertionError):
            _merge_combo_dupls(
                _good_dupls_for_this_combo,
                _poly_dupls_current_partial_fit
            )


    @pytest.mark.parametrize('_poly_dupls_current_partial_fit',
        ([(1,), (2,), (3,)], ([(1,),(2,),(3,)], [4,5,6]), {1,2,3}, [[1,2,3],[]])
    )
    def test_poly_dupls_current_partial_fit_rejects_bad(self, _good_combo_dupls, _poly_dupls_current_partial_fit):
        with pytest.raises(AssertionError):
            _merge_combo_dupls(
                _good_combo_dupls,
                _poly_dupls_current_partial_fit
            )


    @pytest.mark.parametrize('_poly_dupls_current_partial_fit',
        ([], [[(1,),(2,)]], [[(1,),(2,),(3,)], [(4,),(5,),(6,)]])
    )
    def test_poly_dupls_current_partial_fit_accepts_good(self, _good_combo_dupls, _poly_dupls_current_partial_fit):
        _merge_combo_dupls(
            _good_combo_dupls,
            _poly_dupls_current_partial_fit
        )

    # END _poly_dupls_current_partial_fit ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *




class TestMergeComboDuplsAccuracy:


    @pytest.mark.parametrize('_dupls_for_this_combo',
        ([], [(1,)], [(0,), (4,), (5,)], [(0,10), (1,21), (2,33)])
    )
    @pytest.mark.parametrize('_poly_dupls_current_partial_fit',
        (
            [], [[(1,)]], [[(2,)]], [[(0,), (4,), (5,)]], [[(0,), (3,), (5,)]],
            [[(0,10), (1,21), (2,33)]], [[(1,21), (2,34)]]
        )
    )
    def test_accuracy(self, _dupls_for_this_combo, _poly_dupls_current_partial_fit):

        # first pass _dupls_for_this_combo goes directly into _poly_dupls_current_partial_fit
        # if _dupls_for_this_combo is empty, always just returns []

        out = _merge_combo_dupls(
            _dupls_for_this_combo=_dupls_for_this_combo,
            _poly_dupls_current_partial_fit=_poly_dupls_current_partial_fit,
        )


        if not len(_dupls_for_this_combo):
            assert out == _poly_dupls_current_partial_fit
        elif len(_dupls_for_this_combo):
            if not len(_poly_dupls_current_partial_fit):
                assert out == [_dupls_for_this_combo]
            elif len(_poly_dupls_current_partial_fit):
            #     for what in what?!!?!
                pass
















