# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




import pytest

import re

import numpy as np

from pybear.feature_extraction.text._TextReplacer._validation._validation \
    import _validation




class TestValidation:

    # the brunt of the test is done at the individual module level.
    # just test that this takes and passes good args


    @pytest.mark.parametrize('_len', (0, 10, 1000))
    @pytest.mark.parametrize('sr',
        (None, 'tuple_1', 'tuple_2', 'set_1', 'set_2', 'list_1', 'list_2', 'list_3')
    )
    @pytest.mark.parametrize('rr',
        (None, 'tuple_1', 'tuple_2', 'set_1', 'set_2', 'list_1', 'list_2', 'list_3')
    )
    def test_accuracy(self, _len, sr, rr):


        _X = np.random.choice(list('abcdef'), _len, replace=True)

        if sr is None:
            string_replace = None
        elif sr == 'tuple_1':
            string_replace = ('a', '')
        elif sr == 'tuple_2':
            string_replace = ('a', '', 1)
        elif sr == 'set_1':
            string_replace = {('a', ''), ('b', 'B')}
        elif sr == 'set_2':
            string_replace = {('a', '', 2), ('a', '', 1)}
        elif sr == 'list_1':
            string_replace = [('b', 'B') for _ in range(_len)]
        elif sr == 'list_2':
            string_replace = [('b', 'B', 2) for _ in range(_len)]
        elif sr == 'list_3':
            string_replace = [False for _ in range(_len)]
        else:
            raise Exception

        if sr is None:
            regexp_replace = None
        elif sr == 'tuple_1':
            regexp_replace = ('a', '')
        elif sr == 'tuple_2':
            regexp_replace = (re.compile('a'), '', 1)
        elif sr == 'set_1':
            regexp_replace = {('a', ''), (re.compile('b'), 'B')}
        elif sr == 'set_2':
            regexp_replace = {('a', '', 2, re.I), (re.compile('a', re.I), '', 1)}
        elif sr == 'list_1':
            regexp_replace = [('b', 'B', 1, re.X) for _ in range(_len)]
        elif sr == 'list_2':
            regexp_replace = [(re.compile('b', re.I), 'B', 2) for _ in range(_len)]
        elif sr == 'list_3':
            regexp_replace = [False for _ in range(_len)]
        else:
            raise Exception


        _validation(
            _X,
            string_replace,
            regexp_replace
        )






