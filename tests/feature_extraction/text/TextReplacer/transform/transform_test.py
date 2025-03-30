# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from copy import deepcopy
import re

import numpy as np

from pybear.feature_extraction.text._TextReplacer._transform._transform import \
    _transform




class TestTransform:

    # we know that the individual core modules are accurate from their
    # own tests. this module essentially picks the module for the shape
    # of the data, just test that this module returns the right shapes
    # at the right times.


    # def _transform(
    #     _X: XContainer,
    #     _str_replace: StrReplaceType,
    #     _regexp_replace: RegExpReplaceType
    # ) -> XContainer:


    @staticmethod
    @pytest.fixture(scope='module')
    def _1D_text():
        return [
            "Eye of newt and toe of frog,",
            "Wool of bat and tongue of dog",
            "Adder’s fork and blindworm’s sting,"
        ]


    @staticmethod
    @pytest.fixture(scope='module')
    def _2D_text():
        return [
            ["Eye", "of", "newt", "and", "toe", "of", "frog,"],
            ["Wool", "of", "bat", "and", "tongue", "of", "dog"],
            ["Adder’s", "fork", "and", "blindworm’s", "sting,"]
        ]


    @pytest.mark.parametrize('shape', (1, )) # pizza transform() doesnt take 2D any more! 2))
    @pytest.mark.parametrize('mode', ('str', 'regexp', 'both'))
    def test_accuracy(self, shape, mode, _1D_text, _2D_text):

        _1D_start_len = len(_1D_text)
        _2D_start_len = len(_2D_text)

        if shape == 1:
            _X = deepcopy(_1D_text)
        elif shape == 2:
            _X = deepcopy(_2D_text)
        else:
            raise Exception

        if mode == 'str':
            _str_replace = [("Eye of newt and toe of frog,", ""), False, False]
            _regexp_replace = None
        elif mode == 'regexp':
            _str_replace = None
            _regexp_replace = [False, ('.+', '', 1, re.I | re.X), False]
        elif mode == 'both':
            _str_replace = [("Eye of newt and toe of frog,", ""), False, False]
            _regexp_replace = [False, ('.+', '', 1, re.I), False]
        else:
            raise Exception


        out_X = _transform(_X, _str_replace, _regexp_replace)


        if mode == 'str':
            if shape == 1:
                # one string should be replace
                assert isinstance(out_X, list)
                assert all(map(isinstance, out_X, (str for _ in out_X)))
                assert len(out_X) == _1D_start_len

                exp = deepcopy(_1D_text)
                exp[0] = ''
                assert all(map(np.array_equal, out_X, exp))
            elif shape == 2:
                # nothing should be replaced
                assert isinstance(out_X, list)
                for _ in out_X:
                    assert isinstance(_, list)
                    assert all(map(isinstance, _, (str for i in _)))
                assert all(map(np.array_equal, out_X, _2D_text))

        elif mode == 'regexp':
            if shape == 1:
                # one string should be replaced
                assert isinstance(out_X, list)
                assert all(map(isinstance, out_X, (str for _ in out_X)))
                assert len(out_X) == _1D_start_len

                exp = deepcopy(_1D_text)
                exp[1] = ''
                assert all(map(np.array_equal, out_X, exp))
            elif shape == 2:
                # one row should be replaced
                assert isinstance(out_X, list)
                for _ in out_X:
                    assert isinstance(_, list)
                    assert all(map(isinstance, _, (str for i in _)))
                assert len(out_X) == _2D_start_len

                exp = deepcopy(_2D_text)
                exp[1] = ['' for _ in exp[1]]
                assert all(map(np.array_equal, out_X, exp))
        elif mode == 'both':
            if shape == 1:
                # two strings should be replaced
                assert isinstance(out_X, list)
                assert all(map(isinstance, out_X, (str for _ in out_X)))
                assert len(out_X) == _1D_start_len

                exp = deepcopy(_1D_text)
                exp[0] = ''
                exp[1] = ''
                assert all(map(np.array_equal, out_X, exp))
            elif shape == 2:
                # two rows should be replaced
                assert isinstance(out_X, list)
                for _ in out_X:
                    assert isinstance(_, list)
                    assert all(map(isinstance, _, (str for i in _)))
                assert len(out_X) == _2D_start_len

                exp = deepcopy(_2D_text)
                exp[1] = ['' for _ in exp[1]]
                assert all(map(np.array_equal, out_X, exp))
        else:
            raise Exception


