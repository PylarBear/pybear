# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._NGramMerger._transform._slider import _slider

import pytest

import re

import numpy as np



class TestSlider:

    # def _slider(
    #     _line: list[str],
    #     _ngram: Sequence[Union[str, re.Pattern]],
    #     _ngcallable: Union[Callable[[Sequence[str]], str], None],
    #     _sep: Union[str, None]
    # ) -> list[str]:


    @pytest.mark.parametrize('_sep', (None, '@', '&', '__'))
    def test_accuracy_sep(self, _sep):

        _exp_sep = _sep or '_'

        _line1 = ['EGG', 'SANDWICHES', 'AND', 'ICE', 'CREAM']

        _ngram1 = ['EGG', re.compile('sandwich[es]+', re.I)]

        out = _slider(_line1, _ngram1, None, _sep)

        exp = [f'EGG{_exp_sep}SANDWICHES', 'AND', 'ICE', 'CREAM']

        assert np.array_equal(out, exp)

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        _line2 = out.copy()

        _ngram2 = ['ICE', 'CREAM']

        out2 = _slider(_line2, _ngram2, None, _sep)

        exp2 = [f'EGG{_exp_sep}SANDWICHES', 'AND', f'ICE{_exp_sep}CREAM']

        assert np.array_equal(out2, exp2)



    def test_accuracy_callable(self):

        _line1 = ['BIG', 'BIG', 'MONEY', 'NO', 'WHAMMY', 'YES', 'WHAMMY']

        _ngram1 = [re.compile('big', re.I), re.compile('money', re.I)]

        def _callable1(_matches):
            return '__'.join(np.flip(list(_matches)).tolist())

        out = _slider(_line1, _ngram1, _callable1, _sep='(&#(&$)#!(*$')

        exp = ['BIG', 'MONEY__BIG', 'NO', 'WHAMMY', 'YES', 'WHAMMY']

        assert np.array_equal(out, exp)

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        _line2 = out.copy()

        _ngram2 = ['NO', re.compile('WHAMM.+', re.I)]

        def _callable2(_matches):
            return 'BEER&PIZZA'

        out2 = _slider(_line2, _ngram2, _callable2, None)

        exp2 = ['BIG', 'MONEY__BIG', 'BEER&PIZZA', 'YES', 'WHAMMY']

        assert np.array_equal(out2, exp2)


    def test_ignores_empty_line(self):

        out = _slider([], ['NEW', 'YORK'], lambda x: '_'.join(x), None)
        assert isinstance(out, list)
        assert len(out) == 0


    def test_bad_callable(self):

        with pytest.raises(TypeError):

            _line = ['SILLY', 'STRING']

            _slider(_line, _line, lambda x: _line, None)






