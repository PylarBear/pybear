# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._TextJustifier._shared._transform._transform \
    import _transform

import pytest

import re



class TestTransform:

    # we know that _splitter & _stacker are accurate from their own tests,
    # so just do basic checks


    @staticmethod
    @pytest.fixture(scope='function')
    def _text():
        return [
            "THESE are the times that try men’s souls: The summer ",
            "soldier and the sunshine patriot will, in this crisis, shrink ",
            "from the service of his country; but he that stands it now, ",
            "deserves the love and thanks of man and woman."
        ]



    @pytest.mark.parametrize('_n_chars', (80, 100))
    @pytest.mark.parametrize('_sep', (' ', re.compile('[s-t]')))
    @pytest.mark.parametrize('_sep_flags', (None, re.I, re.I | re.X))
    @pytest.mark.parametrize('_line_break', (None, 'l', re.compile('[a-b]')))
    @pytest.mark.parametrize('_line_break_flags', (None, re.I, re.I | re.X))
    @pytest.mark.parametrize('_backfill_sep', (' ', 'zzz'))
    def test_accuracy(
        self, _text, _n_chars, _sep, _sep_flags, _line_break, _line_break_flags,
        _backfill_sep
    ):

        # skip impossible conditions -- -- -- -- -- -- -- -- -- -- -- --

        if isinstance(_sep, re.Pattern) and _sep_flags:
            pytest.skip(reason=f'cant have re.compile and flags')
        if isinstance(_line_break, re.Pattern) and _line_break_flags:
            pytest.skip(reason=f'cant have re.compile and flags')

        # END skip impossible conditions -- -- -- -- -- -- -- -- -- --

        out = _transform(
            _text,
            _n_chars=_n_chars,
            _sep=_sep,
            _sep_flags=_sep_flags,
            _line_break=_line_break,
            _line_break_flags=_line_break_flags,
            _backfill_sep=_backfill_sep
        )


        assert isinstance(out, list)
        assert all(map(isinstance, out, (str for _ in out)))



