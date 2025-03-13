# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import re

from pybear.feature_extraction.text._TextJustifierRegExp._transform._stacker \
    import _stacker

from pybear.feature_extraction.text._TextJustifierRegExp._transform._splitter \
    import _splitter



class TestStacker:


    @staticmethod
    @pytest.fixture(scope='function')
    def _text():
        return [
            "You say you want a Revolution",
            "Well, you know",
            "We all wanna change the world",
            "You tell me that it's evolution",
            "Well, you know",
            "We all wanna change the world",
            "But when you talk about destruction",
            "Don't you know that you can count me out",
            "Don't you know it's gonna be alright",
            "Alright",
            "Alright",
            "You say you got a real solution",
            "Well, you know",
            "We'd all love to see the plan",
            "You ask me",
            "for a contribution",
            "Well, you know",
            "We are doing what we can",
            "But if you want money for people with minds that hate",
            "All I can tell you is brother you have to wait",
            "Don't you know it's gonna be alright",
            "Alright",
            "Alright",
            "You say you'll change the constitution",
            "Well, you know",
            "We all want to change your head",
            "You tell me it's the institution",
            "Well, you know",
            "You'd better free your mind instead",
            "But if you go carrying pictures of Chairman Mao",
            "You ain't going to make it with anyone anyhow",
            "Don't you know it's gonna be alright",
            "Alright",
            "Alright",
            "Alright, alright",
            "Alright, alright",
            "Alright, alright",
            "Alright, alright"
        ]


    @staticmethod
    @pytest.fixture(scope='function')
    def _text_out_of_spitter(_text):

        def foo(_sep, _sep_flags, _line_break, _line_break_flags, n_lines=1000):

            return _splitter(
                _text[:n_lines],
                _sep=_sep,
                _sep_flags=_sep_flags,
                _line_break=_line_break,
                _line_break_flags=_line_break_flags
            )

        return foo


    @pytest.mark.parametrize('_n_chars', (20, 40, 80))
    @pytest.mark.parametrize('_sep', ('\ ', re.compile('(?!x)')))
    @pytest.mark.parametrize('_sep_flags', (None, re.X))
    @pytest.mark.parametrize('_line_break', (' ', re.compile('(?!x)')))
    @pytest.mark.parametrize('_line_break_flags', (None, re.X))
    def test_stacker_doesnt_hang(
        self, _text_out_of_spitter, _n_chars, _sep, _sep_flags,
        _line_break, _line_break_flags
    ):

        # skip impossible -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        if isinstance(_sep, re.Pattern) and _sep_flags:
            pytest.skip(reason=f'cant have re.compile and flags')
        if isinstance(_line_break, re.Pattern) and _line_break_flags:
            pytest.skip(reason=f'cant have re.compile and flags')
        # END skip impossible -- -- -- -- -- -- -- -- -- -- -- -- -- --

        out = _stacker(
            _text_out_of_spitter(_sep, _sep_flags, _line_break, _line_break_flags),
            _n_chars=_n_chars,
            _sep=_sep,
            _sep_flags=_sep_flags,
            _line_break=_line_break,
            _line_break_flags=_line_break_flags,
            _backfill_sep=' '
        )

        assert isinstance(out, list)



    @pytest.mark.parametrize('_n_chars', (20, 40, 80))
    @pytest.mark.parametrize('_sep', ('\ ', ))
    @pytest.mark.parametrize('_sep_flags', (None, ))
    @pytest.mark.parametrize('_line_break', (None, ))
    @pytest.mark.parametrize('_line_break_flags', (None, ))
    def test_accuracy_n_chars(
        self, _text_out_of_spitter, _n_chars, _sep, _sep_flags,
        _line_break, _line_break_flags
    ):

        out = _stacker(
            _text_out_of_spitter(_sep, _sep_flags, _line_break, _line_break_flags),
            _n_chars=_n_chars,
            _sep=_sep,
            _sep_flags=_sep_flags,
            _line_break=_line_break,
            _line_break_flags=_line_break_flags,
            _backfill_sep=' '
        )

        assert isinstance(out, list)
        assert all(map(isinstance, out, (str for _ in out)))

        assert all(map(lambda x: len(x) <= _n_chars, out))
        # the last row might be short, so leave it out
        assert all(map(lambda x: len(x) >= _n_chars-10, out[:-1]))



    @pytest.mark.parametrize('_n_chars', (80, 100))
    @pytest.mark.parametrize('_sep', ('\ ', ))
    @pytest.mark.parametrize('_sep_flags', (None, ))
    @pytest.mark.parametrize('_line_break', ('t', 'e'))
    @pytest.mark.parametrize('_line_break_flags', (re.I,))
    def test_accuracy_line_break(
        self, _text_out_of_spitter, _n_chars, _sep, _sep_flags,
        _line_break, _line_break_flags
    ):

        out = _stacker(
            _text_out_of_spitter(_sep, _sep_flags, _line_break, _line_break_flags),
            _n_chars=_n_chars,
            _sep=_sep,
            _sep_flags=_sep_flags,
            _line_break=_line_break,
            _line_break_flags=_line_break_flags,
            _backfill_sep=' '
        )

        assert isinstance(out, list)
        assert all(map(isinstance, out, (str for _ in out)))
        assert all(map(
            lambda x: x.lower().endswith(list(_line_break)[0].lower()), out[:-2]
        ))


    @pytest.mark.parametrize('_n_chars', (90, 100))
    @pytest.mark.parametrize('_sep', ('\s', ))
    @pytest.mark.parametrize('_sep_flags', (None, ))
    @pytest.mark.parametrize('_line_break', (None, ))
    @pytest.mark.parametrize('_line_break_flags', (None, ))
    @pytest.mark.parametrize('_backfill_sep', ('zzz', 'qqq', 'pyth'))
    def test_accuracy_sep_and_backfill_sep(
        self, _text, _text_out_of_spitter, _n_chars, _sep, _sep_flags,
        _line_break, _line_break_flags, _backfill_sep
    ):

        # only use the first 5 lines

        _split_text = _text_out_of_spitter(
            _sep, _sep_flags, _line_break, _line_break_flags, n_lines=5
        )

        # determine which lines in the original text dont end with _sep
        _no_sep_lines = []
        for _line in _text[:5]:
            if not _line.endswith(list(_sep)[0]):
                _no_sep_lines.append(_line)


        out = _stacker(
            _split_text,
            _n_chars=_n_chars,
            _sep=_sep,
            _sep_flags=_sep_flags,
            _line_break=_line_break,
            _line_break_flags=_line_break_flags,
            _backfill_sep=_backfill_sep
        )

        assert isinstance(out, list)
        assert all(map(isinstance, out, (str for _ in out)))

        # find the locations of the text in the lines that did not end with sep
        # (which looks to be every line)
        # so the ending text for each line should always have _backfill_sep
        # immediately after it.
        # turn the output into one long string
        out = "".join(out)

        for _no_sep_line in _no_sep_lines:
            _idx = out.find(_no_sep_line[-10:]) + 10
            assert out[_idx:_idx+len(_backfill_sep)] == _backfill_sep







