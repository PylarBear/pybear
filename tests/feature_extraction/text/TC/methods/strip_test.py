# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest
import numpy as np

from pybear.feature_extraction.text._TC._methods._strip import _strip




class TestStrip:


    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @pytest.mark.parametrize('junk_X',
        (-2.7, -1, 0, 1, 2.7, True, None, 'trash', [0,1], (1,), {'A':1},
         lambda x: x, [[1,2,3], [4,5,6]])
    )
    def test_rejects_junk_X(self, junk_X):

        with pytest.raises(TypeError):

            _strip(junk_X, True)


    @pytest.mark.parametrize('junk_is_2D',
        (-2.7, -1, 0, 1, 2.7, None, 'trash', [0,1], (1,), {'A':1},
         lambda x: x, [[1,2,3], [4,5,6]])
    )
    def test_rejects_junk_is_2D(self, junk_is_2D):

        with pytest.raises(TypeError):

            _strip(junk_is_2D, True)


    def test_accepts_bool_is_2D(self):

        _strip(list('abcde'), False)

        _strip([['a', 'b'], ['c', 'd']], True)


    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # 1D -- -- -- -- -- -- -- -- -- --

    def test_1D_list_accuracy(self):

        out = _strip(['Sam', 'I', 'am'], False)
        assert isinstance(out, list)
        assert np.array_equal(out, ['Sam', 'I', 'am'])


        out = _strip([' Sam ', ' I ', ' am '], False)
        assert isinstance(out, list)
        assert np.array_equal(out, ['Sam', 'I', 'am'])


        out = _strip(['I   am   Sam', ' Sam ', ' I ', ' am '], False)
        assert isinstance(out, list)
        assert np.array_equal(out, ['I am Sam', 'Sam', 'I', 'am'])


    def test_1D_np_accuracy(self):

        out = _strip(np.array(['Sam', 'I', 'am']), False)
        assert isinstance(out, np.ndarray)
        assert np.array_equal(out, ['Sam', 'I', 'am'])


        out = _strip(np.array([' Sam ', ' I ', ' am ']), False)
        assert isinstance(out, np.ndarray)
        assert np.array_equal(out, ['Sam', 'I', 'am'])


        out = _strip(np.array(['I   am   Sam', ' Sam ', ' I ', ' am ']), False)
        assert isinstance(out, np.ndarray)
        assert np.array_equal(out, ['I am Sam', 'Sam', 'I', 'am'])

    # END 1D -- -- -- -- -- -- -- -- -- --

    # 2D -- -- -- -- -- -- -- -- -- --

    def test_2D_list_accuracy(self):

        out = _strip([['Sam', 'I', 'am'], ['I', 'am', 'Sam']], True)
        assert isinstance(out, list)
        assert all(map(
            np.array_equal,
            out,
            [['Sam', 'I', 'am'], ['I', 'am', 'Sam']]
        ))


        out = _strip([['  Sam  ', 'I   ', '  am'], [' I ', '  am ', ' Sam']], True)
        assert isinstance(out, list)
        assert all(map(
            np.array_equal,
            out,
            [['Sam', 'I', 'am'], ['I', 'am', 'Sam']]
        ))


        out = _strip([['I   am   Sam,  Sam ,  I , am   ']], True)
        assert isinstance(out, list)
        assert np.array_equal(out, [['I am Sam, Sam, I, am']])


    def test_2D_np_accuracy(self):

        out = _strip(np.array([['Sam', 'I', 'am'], ['I', 'am', 'Sam']]), True)
        assert isinstance(out, np.ndarray)
        assert all(map(
            np.array_equal,
            out,
            [['Sam', 'I', 'am'], ['I', 'am', 'Sam']]
        ))


        out = _strip(
            np.array([['  Sam  ', 'I   ', '  am'], [' I ', '  am ', ' Sam']]),
            True
        )
        assert isinstance(out, np.ndarray)
        assert all(map(
            np.array_equal,
            out,
            [['Sam', 'I', 'am'], ['I', 'am', 'Sam']]
        ))


        out = _strip(np.array([['I   am   Sam,  Sam ,  I , am   ']]), True)
        assert isinstance(out, np.ndarray)
        assert np.array_equal(out, [['I am Sam, Sam, I, am']])

    # END 2D -- -- -- -- -- -- -- -- -- --











