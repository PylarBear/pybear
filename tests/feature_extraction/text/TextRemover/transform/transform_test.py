# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from copy import deepcopy

import numpy as np

from pybear.feature_extraction.text._TextRemover._transform._transform import \
    _transform




class TestTransform:

    # we know that the individual core modules are accurate from their
    # own tests. this module essentially picks the module for the shape
    # of the data and str/regexp, just test that this module returns
    # the right shapes at the right times.


    # def _transform(
    #     _X: XContainer,
    #     _str_remove: StrRemoveType,
    #     _regexp_remove: RegExpRemoveType,
    #     _regexp_flags: RegExpFlagsType
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


    @pytest.mark.parametrize('shape', (1, 2))
    @pytest.mark.parametrize('mode', ('str', 'regexp'))
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
            # _str_remove, _regexp_remove, _regexp_flags
            _args = ["Eye of newt and toe of frog,", None, None]
        elif mode == 'regexp':
            # _str_remove, _regexp_remove, _regexp_flags
            _args = [None, [False, '.+', False], None]
        else:
            raise Exception


        out_X, out_mask = _transform(_X, *_args)


        if mode == 'str':
            if shape == 1:
                # one string should be removed
                assert isinstance(out_X, list)
                assert all(map(isinstance, out_X, (str for _ in out_X)))
                assert len(out_X) == _1D_start_len - 1

                assert isinstance(out_mask, np.ndarray)
                assert out_mask.dtype == np.bool_
                assert np.array_equal(out_mask, [False, True, True])

                exp = deepcopy(_1D_text)
                exp.pop(0)
                assert all(map(np.array_equal, out_X, exp))
            elif shape == 2:
                # nothing should be removed
                assert isinstance(out_X, list)
                for _ in out_X:
                    assert isinstance(_, list)
                    assert all(map(isinstance, _, (str for i in _)))
                assert all(map(np.array_equal, out_X, _2D_text))

                assert isinstance(out_mask, np.ndarray)
                assert out_mask.dtype == np.bool_
                assert np.array_equal(out_mask, [True, True, True])

        elif mode == 'regexp':
            if shape == 1:
                # one string should be removed
                assert isinstance(out_X, list)
                assert all(map(isinstance, out_X, (str for _ in out_X)))
                assert len(out_X) == _1D_start_len - 1

                assert isinstance(out_mask, np.ndarray)
                assert out_mask.dtype == np.bool_
                assert np.array_equal(out_mask, [True, False, True])

                exp = deepcopy(_1D_text)
                exp.pop(1)
                assert all(map(np.array_equal, out_X, exp))
            elif shape == 2:
                # one row should be removed
                assert isinstance(out_X, list)
                for _ in out_X:
                    assert isinstance(_, list)
                    assert all(map(isinstance, _, (str for i in _)))
                assert len(out_X) == _2D_start_len - 1

                assert isinstance(out_mask, np.ndarray)
                assert out_mask.dtype == np.bool_
                assert np.array_equal(out_mask, [True, False, True])

                exp = deepcopy(_2D_text)
                exp.pop(1)
                assert all(map(np.array_equal, out_X, exp))



