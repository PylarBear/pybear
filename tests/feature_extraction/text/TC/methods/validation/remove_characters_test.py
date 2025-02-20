# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.feature_extraction.text._TC._methods._validation. \
    _remove_characters import _remove_characters_validation



class TestRemoveCharactersValidation:

    # def _remove_characters_validation(
    #     _allowed_chars: Union[str, None],
    #     _disallowed_chars: Union[str, None]
    # )


    @pytest.mark.parametrize('junk_ac',
        (-2.7, -1, 0, 1, 2.7, True, False, [0,1], (1,), {'a':1}, lambda x: x)
    )
    def test_rejects_junk_allowed_chars(self, junk_ac):

        with pytest.raises(TypeError):

            _remove_characters_validation(junk_ac, None)



    @pytest.mark.parametrize('junk_dc',
        (-2.7, -1, 0, 1, 2.7, True, False, [0,1], (1,), {'a':1}, lambda x: x)
    )
    def test_rejects_junk_allowed_chars(self, junk_dc):

        with pytest.raises(TypeError):

            _remove_characters_validation(None, _disallowed_chars=junk_dc)


    def test_rejects_empty_strings(self):

        with pytest.raises(ValueError):
            _remove_characters_validation('', None)

        with pytest.raises(ValueError):
            _remove_characters_validation(None, '')


    @pytest.mark.parametrize('_ac', ('!@#$%^&*(', None))
    @pytest.mark.parametrize('_dc', ('qwerty', None))
    def test_mix_and_match_strs_and_None(self, _ac, _dc):


        if (_ac is None and _dc is None) or (_ac is not None and _dc is not None):
            with pytest.raises(ValueError):
                _remove_characters_validation(_ac, _disallowed_chars=_dc)
        else:
            out = _remove_characters_validation(_ac, _disallowed_chars=_dc)

            assert out is None







