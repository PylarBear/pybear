# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest
import numpy as np

from pybear.feature_extraction.text._TC._methods._normalize import \
    _normalize



class TestNormalize:

    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @pytest.mark.parametrize('junk_X',
        (-2.7, -1, 0, 1, 2.7, True, None, 'trash', [0, 1], (1,), {'A': 1},
        lambda x: x, [[1, 2, 3], [4, 5, 6]])
    )
    def test_rejects_junk_X(self, junk_X):
        with pytest.raises(TypeError):
            _normalize(junk_X, True, True)


    @pytest.mark.parametrize('junk_is_2D',
        (-2.7, -1, 0, 1, 2.7, None, 'trash', [0, 1], (1,), {'A': 1},
        lambda x: x, [[1, 2, 3], [4, 5, 6]])
    )
    def test_rejects_junk_is_2D(self, junk_is_2D):
        with pytest.raises(TypeError):
            _normalize(list('abcde'), junk_is_2D, True)


    def test_accepts_bool_is_2D(self):

        _normalize(list('abcde'), False, True)

        _normalize([['a', 'b'], ['c', 'd']], True, True)


    @pytest.mark.parametrize('junk_upper',
        (-2.7, -1, 0, 1, 2.7, None, 'trash', [0, 1], (1,), {'A': 1},
        lambda x: x, [[1, 2, 3], [4, 5, 6]])
    )
    def test_rejects_junk_upper(self, junk_upper):
        with pytest.raises(TypeError):
            _normalize(list('abcde'), True, junk_upper)


    def test_accepts_bool_upper(self):
        _normalize(list('abcde'), True, False)

        _normalize([['a', 'b'], ['c', 'd']], True, True)

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # 1D -- -- -- -- -- -- -- -- -- --

    @pytest.mark.parametrize('upper', (True, False))
    def test_1D_list_accuracy(self, upper):

        out = _normalize(['Sam', 'I', 'am'], False, upper)
        assert isinstance(out, list)
        if upper:
            assert np.array_equal(out, ['SAM', 'I', 'AM'])
        elif not upper:
            assert np.array_equal(out, ['sam', 'i', 'am'])

        out = _normalize([' SAM ', ' I ', ' AM '], False, upper)
        assert isinstance(out, list)
        if upper:
            assert np.array_equal(out, [' SAM ', ' I ', ' AM '])
        elif not upper:
            assert np.array_equal(out, [' sam ', ' i ', ' am '])

        out = _normalize(['I   am   Sam', ' Sam ', ' I ', ' am '], False, upper)
        assert isinstance(out, list)
        if upper:
            assert np.array_equal(out, ['I   AM   SAM', ' SAM ', ' I ', ' AM '])
        elif not upper:
            assert np.array_equal(out, ['i   am   sam', ' sam ', ' i ', ' am '])


    @pytest.mark.parametrize('upper', (True, False))
    def test_1D_np_accuracy(self, upper):

        out = _normalize(np.array(['Sam', 'I', 'am']), False, upper)
        assert isinstance(out, np.ndarray)
        if upper:
            assert np.array_equal(out, ['SAM', 'I', 'AM'])
        elif not upper:
            assert np.array_equal(out, ['sam', 'i', 'am'])

        out = _normalize(np.array([' Sam ', ' I ', ' am ']), False, upper)
        assert isinstance(out, np.ndarray)
        if upper:
            assert np.array_equal(out, [' SAM ', ' I ', ' AM '])
        elif not upper:
            assert np.array_equal(out, [' sam ', ' i ', ' am '])

        out = _normalize(
            np.array(['I   am   Sam', ' Sam ', ' I ', ' am ']), False, upper
        )
        assert isinstance(out, np.ndarray)
        if upper:
            assert np.array_equal(out, ['I   AM   SAM', ' SAM ', ' I ', ' AM '])
        elif not upper:
            assert np.array_equal(out, ['i   am   sam', ' sam ', ' i ', ' am '])

    # END 1D -- -- -- -- -- -- -- -- -- --

    # 2D -- -- -- -- -- -- -- -- -- --

    @pytest.mark.parametrize('upper', (True, False))
    def test_2D_list_accuracy(self, upper):

        out = _normalize([['Sam', 'I', 'am'], ['I', 'am', 'Sam']], True, upper)
        assert isinstance(out, list)
        if upper:
            assert all(map(
                np.array_equal,
                out,
                [['SAM', 'I', 'AM'], ['I', 'AM', 'SAM']]
            ))
        elif not upper:
            assert all(map(
                np.array_equal,
                out,
                [['sam', 'i', 'am'], ['i', 'am', 'sam']]
            ))


        out = _normalize(
            [['  Sam  ', 'I   ', '  am'], [' I ', '  am ', ' Sam']], True, upper
        )
        assert isinstance(out, list)
        if upper:
            assert all(map(
                np.array_equal,
                out,
                [['  SAM  ', 'I   ', '  AM'], [' I ', '  AM ', ' SAM']],
            ))
        elif not upper:
            assert all(map(
                np.array_equal,
                out,
                [['  sam  ', 'i   ', '  am'], [' i ', '  am ', ' sam']],
            ))

        out = _normalize([['I   am   Sam,  Sam ,  I , am   ']], True, upper)
        assert isinstance(out, list)
        if upper:
            assert np.array_equal(out, [['I   AM   SAM,  SAM ,  I , AM   ']],)
        elif not upper:
            assert np.array_equal(out, [['i   am   sam,  sam ,  i , am   ']],)


    @pytest.mark.parametrize('upper', (True, False))
    def test_2D_np_accuracy(self, upper):

        out = _normalize(
            np.array([['Sam', 'I', 'am'], ['I', 'am', 'Sam']]), True, upper
        )
        assert isinstance(out, np.ndarray)
        if upper:
            assert all(map(
                np.array_equal,
                out,
                [['SAM', 'I', 'AM'], ['I', 'AM', 'SAM']]
            ))
        elif not upper:
            assert all(map(
                np.array_equal,
                out,
                [['sam', 'i', 'am'], ['i', 'am', 'sam']]
            ))

        out = _normalize(
            np.array([['  Sam  ', 'I   ', '  am'], [' I ', '  am ', ' Sam']]),
            True,
            upper
        )
        assert isinstance(out, np.ndarray)
        if upper:
            assert all(map(
                np.array_equal,
                out,
                [['  SAM  ', 'I   ', '  AM'], [' I ', '  AM ', ' SAM']]
            ))
        elif not upper:
            assert all(map(
                np.array_equal,
                out,
                [['  sam  ', 'i   ', '  am'], [' i ', '  am ', ' sam']]
            ))

        out = _normalize(
            np.array([['I   am   Sam,  Sam ,  I , am   ']]), True, upper
        )
        assert isinstance(out, np.ndarray)

        if upper:
            assert np.array_equal(out, [['I   AM   SAM,  SAM ,  I , AM   ']])
        elif not upper:
            assert np.array_equal(out, [['i   am   sam,  sam ,  i , am   ']])

    # END 2D -- -- -- -- -- -- -- -- -- --









