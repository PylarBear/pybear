# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.preprocessing._InterceptManager._partial_fit._merge_constants import (
    _merge_constants
)



class TestMergeConstantsValidation:

    # old_constants ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('_old_constants',
        (-np.e, -1, 0, 1, np.e, True, False, 'trash', [0,1], (0,1), lambda x: x)
    )
    def test_old_constants_rejects_junk(self, _old_constants):
        with pytest.raises(AssertionError):
            _merge_constants(
                _old_constants,
                {0:1, 1:1},
                _rtol=1e-5,
                _atol=1e-8
            )


    def test_old_constants_rejects_bad(self):
        with pytest.raises(AssertionError):
            # keys must be col idxs
            _merge_constants(
                {'a':0, 'b':1, 'c':np.nan},
                {0:1, 1:1},
                _rtol=1e-5,
                _atol=1e-8
            )


    @pytest.mark.parametrize('_old_constants',
        (None, {}, {0: 1, 4: 0, 5: 0}, {0: 1, 2: np.nan})
    )
    def test_old_constants_accepts_good(self, _old_constants):
        _merge_constants(
            _old_constants,
            {0: 1, 1: 1},
            _rtol=1e-5,
            _atol=1e-8
        )
    # END old_constants ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # new_constants ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('_new_constants',
        (-np.e, -1, 0, 1, np.e, True, False, 'trash', [0,1], (0,1), lambda x: x)
    )
    def test_new_constants_rejects_junk(self, _new_constants):
        with pytest.raises(AssertionError):
            _merge_constants(
                {0:0, 1:1, 10:np.e},
                _new_constants,
                _rtol=1e-5,
                _atol=1e-8
            )


    @pytest.mark.parametrize('_new_constants',
        (None, {'a':0, 'b': 1, 'c':np.nan})
    )
    def test_new_constants_rejects_bad(self, _new_constants):
        # keys must be col idxs
        with pytest.raises(AssertionError):
            _merge_constants(
                {0:0, 1:1, 10:np.e},
                _new_constants,
                _rtol=1e-5,
                _atol=1e-8
            )


    @pytest.mark.parametrize('_new_constants',
        ({}, {0: 1, 4: 0, 5: 0}, {0: 1, 2: np.nan})
    )
    def test_new_constants_accepts_good(self, _new_constants):

        _merge_constants(
            {0: 0, 1: 1, 2: np.nan},
            _new_constants,
            _rtol=1e-5,
            _atol=1e-8
        )

    # END new_constants ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # rtol atol ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    @pytest.mark.parametrize('_junk',
        (None, 'trash', [0,1], (0,1), {0,1}, {'a':1}, lambda x: x)
    )
    def test_rtol_atol_rejects_junk(self, _junk):

        with pytest.raises(AssertionError):
            _merge_constants(
                {0:0, 1:1, 10:np.e},
                {},
                _rtol=_junk,
                _atol=1e-8
            )

        with pytest.raises(AssertionError):
            _merge_constants(
                {0: 0, 1: 1, 10: np.e},
                {},
                _rtol=1e-5,
                _atol=_junk
            )


    @pytest.mark.parametrize('_bad', (-np.e, -1, True, False))
    def test_rtol_atol_rejects_bad(self, _bad):

        with pytest.raises(AssertionError):
            _merge_constants(
                {0: 0, 1: 1, 10: np.e},
                {},
                _rtol=_bad,
                _atol=1e-8
            )

        with pytest.raises(AssertionError):
            _merge_constants(
                {0: 0, 1: 1, 10: np.e},
                {},
                _rtol=1e-5,
                _atol=_bad
            )


    @pytest.mark.parametrize('_good', (0, 1e-5, 1, np.e))
    def test_rtol_atol_accepts_good(self, _good):

        _merge_constants(
            {0: 0, 1: 1, 10: np.e},
            {},
            _rtol=_good,
            _atol=1e-8
        )

        _merge_constants(
            {0: 0, 1: 1, 10: np.e},
            {},
            _rtol=1e-5,
            _atol=_good
        )

    # END rtol atol  ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


class TestMergeConstantsAccuracy:

    @pytest.mark.parametrize('_new_constants',
        ({}, {0:1, 4: 0, 5: 0}, {0: 1, 2: np.nan})
    )
    def test_accuracy_first_pass(self, _new_constants):

        # first pass (_old_constants is None)

        out = _merge_constants(
            _old_constants=None,
            _new_constants=_new_constants,
            _rtol=1e-5,
            _atol=1e-8
        )

        # always just returns _new_constants
        assert out == _new_constants


    @pytest.mark.parametrize('_new_constants',
        ({}, {0:1, 4: 0, 5: 0}, {0: 1, 2: np.nan})
    )
    def test_old_constants_is_empty(self, _new_constants):

        # _old_constants == {}

        out = _merge_constants(
            _old_constants={},
            _new_constants=_new_constants,
            _rtol=1e-5,
            _atol=1e-8
        )

        # always just returns {}
        assert out == {}


    def test_accuracy(self):

        _old_constants = {0:1, 4: 0, 5: 0}

        _rtol_atol = (1e-5, 1e-8)

        # one column dropped out
        _new_constants = {0:1, 5: 0}
        out = _merge_constants(_old_constants, _new_constants, *_rtol_atol)
        assert out == {0: 1, 5: 0}

        # new columns of constants, with overlap
        _new_constants = {0: 1, 2: np.nan, 6: 0}
        out = _merge_constants(_old_constants, _new_constants, *_rtol_atol)
        assert out == {0: 1}

        # new columns of constants, no overlap
        _new_constants = {11: 1, 12: 4, 13: 0}
        out = _merge_constants(_old_constants, _new_constants, *_rtol_atol)
        assert out == {}

        # same columns, value changed
        _new_constants = {0:1, 4: 1, 5:0}
        out = _merge_constants(_old_constants, _new_constants, *_rtol_atol)
        assert out == {0:1, 5:0}

        # empty new constants
        _new_constants = {}
        out = _merge_constants(_old_constants, _new_constants, *_rtol_atol)
        assert out == {}

        # v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^

        # empty old constants --- like on first fit

        OLD_CONSTANTS = (
            {1:1, 3:np.nan},
            {0:1, 6:0},
            {0:1, 4:1, 5:0},
            {7:0, 8:1, 9:2}
        )

        NEW_CONSTANTS = (
            {0: 1, 5: 0},
            {0: 1, 2: np.nan, 6: 0},
            {11: 1, 12: 4, 13: 0},
            {0: 1, 4: 1, 5: 0},
            {}
        )

        for _old_constants in OLD_CONSTANTS:

            for _new_constants in NEW_CONSTANTS:

                out = _merge_constants(
                    _old_constants,
                    _new_constants,
                    *_rtol_atol
                )

                # need to do this the hard way because of np.nan

                exp_out_keys = set(_old_constants).intersection(_new_constants)

                exp_out_dict = {}
                for _key in exp_out_keys:
                    if str(_old_constants[_key]) == str(_new_constants[_key]):
                        exp_out_dict[int(_key)] = _new_constants[_key]

                assert out == exp_out_dict




