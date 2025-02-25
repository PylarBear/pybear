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
        out = _str_2D_core(deepcopy(_text), 'a')
        assert isinstance(out, list)
        assert all(map(isinstance, out, (list for _ in out)))

        exp = deepcopy(_text)
        exp[0].remove('a')
        assert all(map(np.array_equal, out, exp))


        out = _str_2D_core(deepcopy(_text), 't')
        assert isinstance(out, list)
        assert all(map(isinstance, out, (list for _ in out)))

        exp = deepcopy(_text)
        exp[4].remove('t')
        assert all(map(np.array_equal, out, exp))
        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        # two removed
        out = _str_2D_core(deepcopy(_text), {'a', 'q'})
        assert isinstance(out, list)
        assert all(map(isinstance, out, (list for _ in out)))

        exp = deepcopy(_text)
        exp[0].remove('a')
        exp[4].remove('q')
        assert all(map(np.array_equal, out, exp))
        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        # empty row removed
        out = _str_2D_core(deepcopy(_text), set('abcd'))
        assert isinstance(out, list)
        assert all(map(isinstance, out, (list for _ in out)))

        exp = deepcopy(_text)
        exp.pop(0)
        assert all(map(np.array_equal, out, exp))
        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        # multiple matches removed

        _new_text = deepcopy(_text)
        _new_text[0] = list('aaabc')
        _new_text[4] = list('rsttt')

        out = _str_2D_core(deepcopy(_new_text), set('at'))
        assert isinstance(out, list)
        assert all(map(isinstance, out, (list for _ in out)))

        exp = deepcopy(_text)
        exp[0] = ['b', 'c']
        exp[4] = ['r', 's']
        assert all(map(np.array_equal, out, exp))
        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        # list

        out = _str_2D_core(deepcopy(_text), [set('ab'), False, False, False, 'z'])
        assert isinstance(out, list)
        assert all(map(isinstance, out, (list for _ in out)))

        exp = deepcopy(_text)
        exp[0] = list('cd')
        assert all(map(np.array_equal, out, exp))
        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --














