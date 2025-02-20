# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest
import numpy as np

from pybear.feature_extraction.text._TC._methods._remove_characters import \
    _remove_characters




class TestRemoveCharacters:


    # def _remove_characters(
    #     _WIP_X: Union[list[str], list[list[str]], npt.NDArray[str]],
    #     _is_2D: bool,
    #     _allowed_chars: Union[str, None],
    #     _disallowed_chars: Union[str, None]
    # ) -> Union[list[str], list[list[str]], npt.NDArray[str]]:


    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @pytest.mark.parametrize('junk_X',
        (-2.7, -1, 0, 1, 2.7, True, None, 'trash', [0,1], (1,), {'A':1},
         lambda x: x, [[1,2,3], [4,5,6]])
    )
    def test_rejects_junk_X(self, junk_X):

        with pytest.raises(TypeError):

            _remove_characters(junk_X, True, None, '!@#$%^&*()')


    @pytest.mark.parametrize('junk_is_2D',
        (-2.7, -1, 0, 1, 2.7, None, 'trash', [0,1], (1,), {'A':1},
         lambda x: x, [[1,2,3], [4,5,6]])
    )
    def test_rejects_junk_is_2D(self, junk_is_2D):

        with pytest.raises(TypeError):

            _remove_characters(list('abc'), junk_is_2D, None, '!@#$%^&*()')


    def test_accepts_bool_is_2D(self):

        _remove_characters(list('abcde'), False, None, '!@#$%^&*()')

        _remove_characters([['a', 'b'], ['c', 'd']], True, None, '!@#$%^&*()')


    # _allowed & _disallowed validation is tested elsewhere

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # 1D -- -- -- -- -- -- -- -- -- --

    def test_1D_list_accuracy(self):


        out = _remove_characters([' Sam ', ' I ', ' am '], False, None, '!@#')
        assert isinstance(out, list)
        assert np.array_equal(out, [' Sam ', ' I ', ' am '])

        out = _remove_characters(['!S!a!m!', '@I@', '#a#m#'], False, None, '!@#')
        assert isinstance(out, list)
        assert np.array_equal(out, ['Sam', 'I', 'am'])

        out = _remove_characters(['Sam ', ' I ', ' am '], False, 'SamI ', None)
        assert isinstance(out, list)
        assert np.array_equal(out, ['Sam ', ' I ', ' am '])

        out = _remove_characters(
            [' !Sam! ', '@ I @', ' #am #'], False, 'SamI ', None
        )
        assert isinstance(out, list)
        assert np.array_equal(out, [' Sam ', ' I ', ' am '])


        # removes empties
        out = _remove_characters(
            [' !Sam! ', '@ I @', ' #am #'], False, None, ' #am'
        )
        assert isinstance(out, list)
        assert np.array_equal(out, ['!S!', '@I@'])


    def test_1D_np_accuracy(self):

        out = _remove_characters(
            np.array([' Sam ', ' I ', ' am ']), False, None, '!@#'
        )
        assert isinstance(out, np.ndarray)
        assert np.array_equal(out, [' Sam ', ' I ', ' am '])

        out = _remove_characters(
            np.array(['!S!a!m!', '@I@', '#a#m#']), False, None, '!@#'
        )
        assert isinstance(out, np.ndarray)
        assert np.array_equal(out, ['Sam', 'I', 'am'])

        out = _remove_characters(
            np.array(['Sam ', ' I ', ' am ']), False, 'SamI ', None
        )
        assert isinstance(out, np.ndarray)
        assert np.array_equal(out, ['Sam ', ' I ', ' am '])

        out = _remove_characters(
            np.array([' !Sam! ', '@ I @', ' #am #']), False, 'SamI ', None
        )
        assert isinstance(out, np.ndarray)
        assert np.array_equal(out, [' Sam ', ' I ', ' am '])

        # removes empties
        out = _remove_characters(
            np.array([' !Sam! ', '@ I @', ' #am #']), False, None, ' #am'
        )
        assert isinstance(out, np.ndarray)
        assert np.array_equal(out, ['!S!', '@I@'])


    # END 1D -- -- -- -- -- -- -- -- -- --

    # 2D -- -- -- -- -- -- -- -- -- --

    def test_2D_list_accuracy(self):

        out = _remove_characters(
            [['Sam', 'I', 'am'], ['I', 'am', 'Sam']],
            True,
            None,
            '!@#'
        )
        assert isinstance(out, list)
        assert all(map(
            np.array_equal,
            out,
            [['Sam', 'I', 'am'], ['I', 'am', 'Sam']]
        ))

        # -- -- -- -- -- -- -- -- -- -- -- -- -- --

        out = _remove_characters(
            [[' !Sam! ', 'I@  ', ' #am'], [' !I@ ', ' @am#', '@Sam']],
            True,
            None,
            '!@#'
        )
        assert isinstance(out, list)
        assert all(map(
            np.array_equal,
            out,
            [[' Sam ', 'I  ', ' am'], [' I ', ' am', 'Sam']]
        ))

        # -- -- -- -- -- -- -- -- -- -- -- -- -- --

        out = _remove_characters(
            [['I   am   Sam,  Sam ,  I , am   ']],
            True,
            'IamS ',
            None
        )
        assert isinstance(out, list)
        assert np.array_equal(out, [['I   am   Sam  Sam   I  am   ']])

        # -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # removes empties
        out = _remove_characters(
            [[' !Sam! ', 'I@  ', ' #am'], [' !I@ ', ' @am#', '@Sam']],
            True,
            None,
            '@Sam'
        )
        assert isinstance(out, list)
        assert all(map(
            np.array_equal,
            out,
            [[' !! ', 'I  ', ' #'], [' !I ', ' #']]
        ))


    def test_2D_np_accuracy(self):

        out = _remove_characters(
            np.array([['Sam', 'I', 'am'], ['I', 'am', 'Sam']]),
            True,
            None,
            '!@#'
        )
        assert isinstance(out, np.ndarray)
        assert all(map(
            np.array_equal,
            out,
            [['Sam', 'I', 'am'], ['I', 'am', 'Sam']]
        ))

        # -- -- -- -- -- -- -- -- -- -- -- -- -- --

        out = _remove_characters(
            np.array([['!S!a!m!', '@I@', '#a#m#'], ['!I ', ' @am ', ' Sam#']]),
            True,
            None,
            '!@#'
        )
        assert isinstance(out, np.ndarray)
        assert all(map(
            np.array_equal,
            out,
            [['Sam', 'I', 'am'], ['I ', ' am ', ' Sam']]
        ))

        # -- -- -- -- -- -- -- -- -- -- -- -- -- --

        out = _remove_characters(
            np.array([['I   am   Sam,  Sam ,  I , am   ']]),
            True,
            'IamS ',
            None
        )
        assert isinstance(out, np.ndarray)
        assert np.array_equal(out, [['I   am   Sam  Sam   I  am   ']])


        # removes empties
        out = _remove_characters(
            np.array(
                (np.array([' !Sam! ', 'I@  ']),
                np.array([' !I@ ', ' @am#', '@Sam'])),
                dtype=object
            ),
            True,
            None,
            '@Sam'
        )
        assert isinstance(out, np.ndarray)
        assert all(map(
            np.array_equal,
            out,
            [[' !! ', 'I  '], [' !I ', ' #']]
        ))

        # this is actually a full array so will skip removing the empties
        # because of casting error
        out = _remove_characters(
            np.array([[' !Sam! ', 'I@  ', ' #am'], [' !I@ ', ' @am#', '@Sam']]),
            True,
            None,
            '@Sam'
        )
        assert isinstance(out, np.ndarray)
        assert all(map(
            np.array_equal,
            out,
            [[' !! ', 'I  ', ' #'], [' !I ', ' #', '']]
        ))

    # END 2D -- -- -- -- -- -- -- -- -- --











