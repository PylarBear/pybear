# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from copy import deepcopy

import numpy as np

from pybear.feature_extraction.text._TextRemover._transform._str_2D_core import \
    _str_2D_core



class TestStr2DCore:


    @staticmethod
    @pytest.fixture(scope='module')
    def _text():
        return [
            list('abcd'),
            list('efgh'),
            list('ijkl'),
            list('mnop'),
            list('qrst')
        ]

    def test_accuracy(self, _text):

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        # one removed
        out_X, out_mask = _str_2D_core(deepcopy(_text), 'a')

        assert isinstance(out_X, list)
        assert all(map(isinstance, out_X, (list for _ in out_X)))

        assert isinstance(out_mask, np.ndarray)
        assert out_mask.dtype == np.bool_
        assert np.array_equal(out_mask, [True, True, True, True, True])

        exp = deepcopy(_text)
        exp[0].remove('a')
        assert all(map(np.array_equal, out_X, exp))


        out_X, out_mask = _str_2D_core(deepcopy(_text), 't')

        assert isinstance(out_X, list)
        assert all(map(isinstance, out_X, (list for _ in out_X)))

        assert isinstance(out_mask, np.ndarray)
        assert out_mask.dtype == np.bool_
        assert np.array_equal(out_mask, [True, True, True, True, True])

        exp = deepcopy(_text)
        exp[4].remove('t')
        assert all(map(np.array_equal, out_X, exp))
        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        # two removed
        out_X, out_mask = _str_2D_core(deepcopy(_text), {'a', 'q'})
        assert isinstance(out_X, list)
        assert all(map(isinstance, out_X, (list for _ in out_X)))

        assert isinstance(out_mask, np.ndarray)
        assert out_mask.dtype == np.bool_
        assert np.array_equal(out_mask, [True, True, True, True, True])

        exp = deepcopy(_text)
        exp[0].remove('a')
        exp[4].remove('q')
        assert all(map(np.array_equal, out_X, exp))
        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        # empty row removed
        out_X, out_mask = _str_2D_core(deepcopy(_text), set('abcd'))
        assert isinstance(out_X, list)
        assert all(map(isinstance, out_X, (list for _ in out_X)))

        assert isinstance(out_mask, np.ndarray)
        assert out_mask.dtype == np.bool_
        assert np.array_equal(out_mask, [False, True, True, True, True])

        exp = deepcopy(_text)
        exp.pop(0)
        assert all(map(np.array_equal, out_X, exp))
        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        # multiple matches removed

        _new_text = deepcopy(_text)
        _new_text[0] = list('aaabc')
        _new_text[4] = list('rsttt')

        out_X, out_mask = _str_2D_core(deepcopy(_new_text), set('at'))
        assert isinstance(out_X, list)
        assert all(map(isinstance, out_X, (list for _ in out_X)))

        assert isinstance(out_mask, np.ndarray)
        assert out_mask.dtype == np.bool_
        assert np.array_equal(out_mask, [True, True, True, True, True])

        exp = deepcopy(_text)
        exp[0] = ['b', 'c']
        exp[4] = ['r', 's']
        assert all(map(np.array_equal, out_X, exp))
        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        # list

        out_X, out_mask = \
            _str_2D_core(deepcopy(_text), [set('ab'), False, False, False, 'z'])
        assert isinstance(out_X, list)
        assert all(map(isinstance, out_X, (list for _ in out_X)))

        assert isinstance(out_mask, np.ndarray)
        assert out_mask.dtype == np.bool_
        assert np.array_equal(out_mask, [True, True, True, True, True])

        exp = deepcopy(_text)
        exp[0] = list('cd')
        assert all(map(np.array_equal, out_X, exp))
        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --














