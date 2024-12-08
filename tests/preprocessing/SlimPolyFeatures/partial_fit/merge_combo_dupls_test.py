# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from pybear.preprocessing.SlimPolyFeatures._partial_fit._merge_combo_dupls import (
    _merge_combo_dupls
)

import numpy as np
from copy import deepcopy

import pytest




class TestMergeComboDupls:


    @staticmethod
    @pytest.fixture(scope='module')
    def _good_combo_dupls():
        return [(1,), (1,2)]


    @staticmethod
    @pytest.fixture(scope='module')
    def _good_partialfit_dupls():
        return [[(1, ), (0, 2)]]



    # _dupls_for_this_combo ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    @pytest.mark.parametrize('_dupls_for_this_combo',
        (-np.e, -1, 0, 1, np.e, None, True, False, 'trash', lambda x: x)
    )
    def test_dupls_for_this_combo_rejects_junk(
        self, _dupls_for_this_combo, _good_partialfit_dupls
    ):

        with pytest.raises(AssertionError):
            _merge_combo_dupls(
                _dupls_for_this_combo,
                _good_partialfit_dupls
            )


    @pytest.mark.parametrize('_dupls_for_this_combo',
        ((), [(0,)], ((0,), (1,)), {(0,), (1,)}, [[0], [1]])
    )
    def test_dupls_for_this_combo_rejects_bad(
        self, _dupls_for_this_combo, _good_partialfit_dupls
    ):

        # must be list(tuple[int)), len != 1

        with pytest.raises(AssertionError):
            _merge_combo_dupls(
                _dupls_for_this_combo,
                _good_partialfit_dupls
            )


    @pytest.mark.parametrize('_dupls_for_this_combo',
        ([(0,), (1,), (2,)], [(0,), (1,), (0,1)])
    )
    def test_dupls_for_this_combo_rejects_bad_last_len(
        self, _dupls_for_this_combo, _good_partialfit_dupls
    ):
        # dupls_for_this_combo_ can have any number of single tuple slots
        # (number of dupls found in X), but the last slot must always be
        # len(tuple) >= 2 (the last slot must always be an idx combo)
        with pytest.raises(AssertionError):
            _merge_combo_dupls(
                _dupls_for_this_combo=_dupls_for_this_combo,
                _poly_dupls_current_partial_fit=_good_partialfit_dupls
            )


    @pytest.mark.parametrize('_dupls_for_this_combo',
        ([(0,), (0,), (0,1)], [(0,), (0,1), (0,1)])
    )
    def test_dupls_for_this_combo_rejects_not_all_unique(
        self, _dupls_for_this_combo, _good_partialfit_dupls
    ):

        with pytest.raises(AssertionError):
            _merge_combo_dupls(
                _dupls_for_this_combo=_dupls_for_this_combo,
                _poly_dupls_current_partial_fit=_good_partialfit_dupls
            )


    @pytest.mark.parametrize('_dupls_for_this_combo',
        ([], [(0,), (1,2)], [(0,), (2,3)], [(2,), (0,1,2)])
    )
    def test_dupls_for_this_combo_accepts_good(
        self, _dupls_for_this_combo, _good_partialfit_dupls
    ):

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
        self, _good_combo_dupls, _poly_dupls_current_partial_fit
    ):

        with pytest.raises(AssertionError):
            _merge_combo_dupls(
                _good_combo_dupls,
                _poly_dupls_current_partial_fit
            )


    @pytest.mark.parametrize('_poly_dupls_current_partial_fit',
        ([(1,), (2,), (3,)], ([(1,),(2,),(3,)], [4,5,6]), {1,2,3}, [[1,2,3],[]])
    )
    def test_poly_dupls_current_partial_fit_rejects_bad(
        self, _good_combo_dupls, _poly_dupls_current_partial_fit
    ):

        with pytest.raises(AssertionError):
            _merge_combo_dupls(
                _good_combo_dupls,
                _poly_dupls_current_partial_fit
            )


    @pytest.mark.parametrize('_poly_dupls_current_partial_fit',
        ([[(0,), (1,), (2,)]], [[(0,), (1,), (0,1)]])
    )
    def test_dupls_for_this_combo_rejects_bad_last_len(
        self, _good_combo_dupls, _poly_dupls_current_partial_fit
    ):
        # dupls_for_this_combo_ can have any number of single tuple slots
        # (number of dupls found in X), but the last slot must always be
        # len(tuple) >= 2 (the last slot must always be an idx combo)
        if len(_poly_dupls_current_partial_fit[0][-1]) < 2:
            with pytest.raises(AssertionError):
                _merge_combo_dupls(
                    _dupls_for_this_combo=_good_combo_dupls,
                    _poly_dupls_current_partial_fit=_poly_dupls_current_partial_fit
                )
        else:
            _merge_combo_dupls(
                _dupls_for_this_combo=_good_combo_dupls,
                _poly_dupls_current_partial_fit=_poly_dupls_current_partial_fit
            )


    @pytest.mark.parametrize('_partialfit_dupls',
        ([[(0,), (1,), (1,)]], [[(0,), (0,1)], [(1,), (0,1)]])
    )
    def test_dupls_for_this_combo_rejects_not_all_unique(
        self, _good_combo_dupls, _partialfit_dupls
    ):

        with pytest.raises(AssertionError):
            _merge_combo_dupls(
                _dupls_for_this_combo=_good_combo_dupls,
                _poly_dupls_current_partial_fit=_partialfit_dupls
            )


    @pytest.mark.parametrize('_poly_dupls_current_partial_fit',
        ([], [[(1,),(1,2)]], [[(1,),(2,),(1,2)], [(4,),(5,),(4,7)]])
    )
    def test_poly_dupls_current_partial_fit_accepts_good(
        self, _good_combo_dupls, _poly_dupls_current_partial_fit
    ):

        _merge_combo_dupls(
            _good_combo_dupls,
            _poly_dupls_current_partial_fit
        )

    # END _poly_dupls_current_partial_fit ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *






class TestMergeComboDuplsAccuracy:


    @pytest.mark.parametrize('_dupls_for_this_combo',
        ([], [(0,), (4,), (1,2)], [(0,), (2,3)])
    )
    def test_accuracy_first_pass(self, _dupls_for_this_combo):

        # first partial_fit (_poly_dupls_current_partial_fit is empty)

        out = _merge_combo_dupls(
            _dupls_for_this_combo=_dupls_for_this_combo,
            _poly_dupls_current_partial_fit=[]
        )

        if not len(_dupls_for_this_combo):
            assert out == []
        else:
            # always just returns _dupls_for_this_combo if there are any
            assert out == [_dupls_for_this_combo]



    @pytest.mark.parametrize('_poly_dupls_current_partial_fit',
        ([], [[(0,), (4,), (1,2)]], [[(2,), (2,3)]])
    )
    def test_dupls_for_this_combo_is_empty(self, _poly_dupls_current_partial_fit):

        # _dupls_for_this_combo=[], _poly_dupls_current_partial_fit always
        # returned unchanged.

        out = _merge_combo_dupls(
            _dupls_for_this_combo=[],
            _poly_dupls_current_partial_fit=_poly_dupls_current_partial_fit
        )

        assert out == _poly_dupls_current_partial_fit



    def test_accuracy(self):

        # first idx not in _poly_dupls_current_partial_fit,
        # add entire _dupls_for_this_combo to _poly_dupls_current_partial_fit
        _dupls_for_this_combo = [(2, ), (2,3)]
        _poly_dupls_current_partial_fit = [
            [(0,), (1, 2)],
            [(1,), (0, 2)]
        ]
        out = _merge_combo_dupls(
            _dupls_for_this_combo=_dupls_for_this_combo,
            _poly_dupls_current_partial_fit=_poly_dupls_current_partial_fit
        )

        exp = deepcopy(_poly_dupls_current_partial_fit)
        exp.append(_dupls_for_this_combo)

        assert out == exp

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # first idx of _dupls_for_this_combo already in _poly_dupls_current_partial_fit,
        # merge the two sets that share the first idx

        # new columns of constants, with overlap
        _dupls_for_this_combo = [(1, ), (2,3)]
        _poly_dupls_current_partial_fit = [
            [(0,), (1, 2)],
            [(1,), (0, 2)]
        ]
        out = _merge_combo_dupls(
            _dupls_for_this_combo=_dupls_for_this_combo,
            _poly_dupls_current_partial_fit=_poly_dupls_current_partial_fit
        )

        exp = deepcopy(_poly_dupls_current_partial_fit)
        exp[1].append(_dupls_for_this_combo[-1])

        assert out == exp












