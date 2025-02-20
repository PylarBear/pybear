# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.feature_extraction.text._TC._methods._validation._menu \
    import _menu_validation



class TestMenuValidation:


    @staticmethod
    @pytest.fixture(scope='module')
    def _menu_keys():
        return 'ABCD'


    @pytest.mark.parametrize('junk_possible',
        (-2.7, -1, 0, 1, 2.7, True, False, [0,1], (1,), {'a':1}, lambda x: x)
    )
    def test_rejects_junk_possible(self, junk_possible):

        with pytest.raises(TypeError):

            _menu_validation(
                junk_possible,
                _allowed=None,
                _disallowed=None
            )


    @pytest.mark.parametrize('junk_allowed',
        (-2.7, -1, 0, 1, 2.7, True, False, [0,1], (1,), {'a':1}, lambda x: x)
    )
    def test_rejects_junk_allowed(self, junk_allowed, _menu_keys):

        with pytest.raises(TypeError):

            _menu_validation(
                _menu_keys,
                _allowed=junk_allowed,
                _disallowed=None
            )


    @pytest.mark.parametrize('junk_disallowed',
        (-2.7, -1, 0, 1, 2.7, True, False, [0,1], (1,), {'a':1}, lambda x: x)
    )
    def test_rejects_junk_disallowed(self, junk_disallowed, _menu_keys):

        with pytest.raises(TypeError):

            _menu_validation(
                _menu_keys,
                _allowed=None,
                _disallowed=junk_disallowed
            )


    def test_rejects_empty_strings(self):

        with pytest.raises(ValueError):

            _menu_validation(
                '',
                _allowed=None,
                _disallowed=None
            )


        with pytest.raises(ValueError):

            _menu_validation(
                'ABCD',
                _allowed='',
                _disallowed=None
            )


        with pytest.raises(ValueError):

            _menu_validation(
                'ABCD',
                _allowed=None,
                _disallowed=''
            )


    @pytest.mark.parametrize('_allowed', (None, 'AB'))
    @pytest.mark.parametrize('_disallowed', (None, 'CD'))
    def test_mix_and_match_None(self, _allowed, _disallowed, _menu_keys):

        if _allowed is not None and _disallowed is not None:

            with pytest.raises(ValueError):
                _menu_validation(
                    _menu_keys,
                    _allowed=_allowed,
                    _disallowed=_disallowed
                )

        else:

            out = _menu_validation(
                _menu_keys,
                _allowed=_allowed,
                _disallowed=_disallowed
            )

            assert out is None


    @pytest.mark.parametrize('impossible_char', list('abcdwxyz'))
    def test_rejects_impossible_char(self, impossible_char, _menu_keys):


        allowed = 'abc' + impossible_char

        with pytest.raises(ValueError):
            _menu_validation(
                _menu_keys,
                _allowed=allowed,
                _disallowed=None
            )


        disallowed = 'abc' + impossible_char

        with pytest.raises(ValueError):
            _menu_validation(
                _menu_keys,
                _allowed=None,
                _disallowed=disallowed

            )


    @pytest.mark.parametrize('possible_char', list('ABCD'))
    def test_accepts_possible_char(self, possible_char, _menu_keys):


        _menu_validation(
            _menu_keys,
            _allowed=possible_char,
            _disallowed=None
        )


        _menu_validation(
            _menu_keys,
            _allowed=None,
            _disallowed=possible_char
        )







