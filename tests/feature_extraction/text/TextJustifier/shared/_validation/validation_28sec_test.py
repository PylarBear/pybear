# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._TextJustifier._shared. _validation. \
    _validation import _validation


import pytest

import re



# the brunt of the validation is handled by the individual modules.
# prove out that this accepts and passes good values in 'str' and 'regex' modes
# for 'str' and 'regex' modes, prove out that join_2D is ignored when data is 1D
# for 'str' mode, prove out conditional reject of 'sep' and 'line_break'



class Fixtures:


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



class TestStrValidation(Fixtures):


    @pytest.mark.parametrize('_X_dim', (1, 2))
    @pytest.mark.parametrize('_sep', (' ', '  ', ',', {'r', ' ', 't'}))
    @pytest.mark.parametrize('_line_break', (' ', 'q', {'_', ' ', ','}))
    def test_sep_linebreak_conditional(
        self, _X_dim, _1D_X, _2D_X, _sep, _line_break
    ):


        if _X_dim == 1:
            _X_wip = _1D_X
        elif _X_dim == 2:
            _X_wip = _2D_X
        else:
            raise Exception

        _n_chars = 100
        _case_sensitive = False
        _sep_flags = None
        _line_break_flags = None
        _join_2D = ' '
        _backfill_sep = ' '
        args = ('str', _X_wip, _n_chars, _sep, _sep_flags, _line_break,
                _line_break_flags, _case_sensitive, _backfill_sep, _join_2D)

        _value_error = False
        _sep = {_sep, } if isinstance(_sep, str) else _sep
        if _line_break is None:
            _line_break = set()
        elif isinstance(_line_break, str):
            _line_break = {_line_break, }
        _union = _sep | _line_break

        if len(_sep) == 0 or len(_line_break) == 0:
            pass
        elif len(_union) != len(_sep) + len(_line_break):
            _value_error = True
        else:
            for s1 in _union:
                if any(s1 in s2 for s2 in _union if s2 != s1):
                    _value_error = True

        if _value_error:
            with pytest.raises(ValueError):
                _validation(*args)
        else:
            assert _validation(*args) is None



    @pytest.mark.parametrize('_X_dim', (1, 2))
    @pytest.mark.parametrize('_n_chars', (1, 10, 100))
    @pytest.mark.parametrize('_sep', ('a', {'b', 'c'}))
    @pytest.mark.parametrize('_sep_flags', (re.I, re.X))
    @pytest.mark.parametrize('_line_break', ('d', {'e', 'f'}))
    @pytest.mark.parametrize('_line_break_flags', (re.I, re.X))
    @pytest.mark.parametrize('_case_sensitive', (True, False))
    @pytest.mark.parametrize('_backfill_sep', (' ', ',', ''))
    @pytest.mark.parametrize('_join_2D', (' ', 1, [0,1], lambda x: x))
    def test_str_accepts_good(
        self, _X_dim, _1D_X, _2D_X, _n_chars, _sep, _sep_flags, _line_break,
        _line_break_flags, _case_sensitive, _backfill_sep, _join_2D
    ):

        args = (_n_chars, _sep, _sep_flags, _line_break, _line_break_flags,
                _case_sensitive, _backfill_sep, _join_2D)

        if _X_dim == 1:
            _X_wip = _1D_X
            # join_2D can be anything

            out = _validation('str', _X_wip, *args)
            assert out is None
        elif _X_dim == 2:
            _X_wip = _2D_X
            if not isinstance(_join_2D, str):
                with pytest.raises(TypeError):
                    _validation('str', _X_wip, *args)
            else:
                out = _validation('str', _X_wip, *args)
                assert out is None
        else:
            raise Exception



class TestRegexValidation(Fixtures):


    @pytest.mark.parametrize('_X_dim', (1, 2))
    @pytest.mark.parametrize('_n_chars', (1, 100))
    @pytest.mark.parametrize('_sep', (re.compile('\a'), re.compile('c')))
    @pytest.mark.parametrize('_sep_flags', (re.I, re.X))
    @pytest.mark.parametrize('_line_break', (re.compile('d'), re.compile('b')))
    @pytest.mark.parametrize('_line_break_flags', (re.I, re.X))
    @pytest.mark.parametrize('_case_sensitive', (True, False))
    @pytest.mark.parametrize('_backfill_sep', (' ', ','))
    @pytest.mark.parametrize('_join_2D', (' ', 1, [0,1], lambda x: x))
    def test_regex_accepts_good(
        self, _X_dim, _1D_X, _2D_X, _n_chars, _sep, _sep_flags, _line_break,
        _line_break_flags, _case_sensitive, _backfill_sep, _join_2D
    ):

        args = (_n_chars, _sep, _sep_flags, _line_break, _line_break_flags,
                _case_sensitive, _backfill_sep, _join_2D)

        if _X_dim == 1:
            _X_wip = _1D_X
            # join_2D can be anything

            out = _validation('regex', _X_wip, *args)
            assert out is None
        elif _X_dim == 2:
            _X_wip = _2D_X
            if not isinstance(_join_2D, str):
                with pytest.raises(TypeError):
                    _validation('regex', _X_wip, *args)
            else:
                out = _validation('regex', _X_wip, *args)
                assert out is None
        else:
            raise Exception










