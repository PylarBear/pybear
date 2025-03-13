# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._TextJustifierRegExp._validation._validation \
    import _validation


import pytest

import re



class TestValidation:

    # the brunt of the validation is handled by the individual modules.
    # prove out that this accepts and passes good values
    # prove out that join_2D is ignored when data is 1D
    # prove out conditional reject of 'sep' and 'line_break'


    @staticmethod
    @pytest.fixture(scope='function')
    def _1D_X():
        return [
            'Make that cat go away!',
            'Tell that Cat in the Hat',
            'You do not want to play.',
            'He should not be here.'
        ]


    @staticmethod
    @pytest.fixture(scope='function')
    def _2D_X():
        return [
            ['Make', 'that', 'cat', 'go', 'away!'],
            ['Tell', 'that', 'Cat', 'in', 'the', 'Hat'],
            ['You', 'do', 'not', 'want', 'to', 'play.'],
            ['He', 'should', 'not', 'be', 'here.']
        ]


    # END fixtures -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --



    @pytest.mark.parametrize('_X_dim', (1, 2))
    @pytest.mark.parametrize('_n_chars', (1, 100))
    @pytest.mark.parametrize('_sep', ('a', re.compile('a')))
    @pytest.mark.parametrize('_sep_flags', (re.I, re.X))
    @pytest.mark.parametrize('_line_break', ('d', re.compile('b')))
    @pytest.mark.parametrize('_line_break_flags', (re.I, re.X))
    @pytest.mark.parametrize('_backfill_sep', (' ', ','))
    @pytest.mark.parametrize('_join_2D', (' ', 1, [0,1], lambda x: x))
    def test_accepts_good(
        self, _X_dim, _1D_X, _2D_X, _n_chars, _sep, _sep_flags, _line_break,
        _backfill_sep, _line_break_flags, _join_2D
    ):

        args = (_n_chars, _sep, _sep_flags, _line_break, _line_break_flags,
                _backfill_sep, _join_2D)

        if _X_dim == 1:
            _X_wip = _1D_X
            # join_2D can be anything

            out = _validation(_X_wip, *args)
            assert out is None
        elif _X_dim == 2:
            _X_wip = _2D_X
            if not isinstance(_join_2D, str):
                with pytest.raises(TypeError):
                    _validation(_X_wip, *args)
            else:
                out = _validation(_X_wip, *args)
                assert out is None
        else:
            raise Exception










