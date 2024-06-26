# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import io
from unittest.mock import patch

from pybear.feature_extraction.text._TextCleaner._lex_lookup._word_editor \
    import _word_editor


class TestWordEditor:

    @pytest.mark.parametrize('bad_prompt',
        (0, 1, 3.14, True, None, min, [1,2], (1,2), {1,2}, {'a':1}, lambda x: x)
    )
    def test_rejects_non_str_prompt(self, bad_prompt):

        with pytest.raises(TypeError):
            _word_editor(bad_prompt)


    def test_accuracy(self):

        for user_input in ('apple\nY\n', 'banana\nY\n', 'cherry\nY\n',
                          'LEMON\nY\n', 'orange\nY\n'):

            with patch('sys.stdin', io.StringIO(user_input)):
                out = _word_editor(user_input)

                assert isinstance(out, str)
                assert out == user_input[:user_input.find('\n')]






