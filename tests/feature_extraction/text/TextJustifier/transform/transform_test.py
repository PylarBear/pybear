# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._TextJustifier._transform._transform \
    import _transform

import pytest



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
    @pytest.mark.parametrize('_sep', (' ', {'s', 't'}))
    @pytest.mark.parametrize('_line_break', ('l', {'a', 'b'}))
    @pytest.mark.parametrize('_backfill_sep', (' ', 'zzz'))
    def test_accuracy(self, _text, _n_chars, _sep, _line_break, _backfill_sep):


        out = _transform(
            _text,
            _n_chars=_n_chars,
            _sep=_sep,
            _line_break=_line_break,
            _backfill_sep=_backfill_sep
        )


        assert isinstance(out, list)
        assert all(map(isinstance, out, (str for _ in out)))



