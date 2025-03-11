# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._TextJustifier._validation._validation import \
    _validation


import pytest



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
    @pytest.mark.parametrize('_sep', (' ', ',', {'r', ' ', 't'}))
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
        _join_2D = ' '
        _backfill_sep = ' '
        args = (_X_wip, _n_chars, _sep, _line_break, _backfill_sep, _join_2D)

        _value_error = False
        if isinstance(_sep, str):
            if isinstance(_line_break, str):
                if _line_break in _sep:
                    _value_error = True
            else:
                for _lb in _line_break:
                    if _lb in _sep:
                        _value_error = True
        elif isinstance(_sep, set):
            if isinstance(_line_break, str):
                for _s in _sep:
                    if _line_break in _s:
                        _value_error = True
            else:
                # both must be sets
                for _lb in _line_break:
                    for _s in _sep:
                        if _lb in _s:
                            _value_error = True

        if _value_error:
            with pytest.raises(ValueError):
                _validation(*args)
        else:
            out = _validation(*args)
            assert out is None


    @pytest.mark.parametrize('_X_dim', (1, 2))
    @pytest.mark.parametrize('_n_chars', (1, 10, 100))
    @pytest.mark.parametrize('_sep', ('a', {'b', 'c'}))
    @pytest.mark.parametrize('_line_break', ('d', {'e', 'f'}))
    @pytest.mark.parametrize('_backfill_sep', (' ', ',', ''))
    @pytest.mark.parametrize('_join_2D', (' ', 1, [0,1], lambda x: x))
    def test_accepts_good(
        self, _X_dim, _1D_X, _2D_X, _n_chars, _sep, _line_break, _backfill_sep,
        _join_2D
    ):

        args = (_n_chars, _sep, _line_break, _backfill_sep, _join_2D)

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










