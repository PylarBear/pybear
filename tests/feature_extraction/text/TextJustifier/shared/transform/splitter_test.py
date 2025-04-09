# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import re

import numpy as np

from pybear.feature_extraction.text._TextJustifier._shared._transform. \
    _splitter import _splitter



class TestSplitter:

    # no validation



    @pytest.mark.parametrize('_sep', (' ', re.compile('(?!x)')))
    @pytest.mark.parametrize('_sep_flags', (None, re.X))
    @pytest.mark.parametrize('_line_break', (' ', re.compile('(?!x)')))
    @pytest.mark.parametrize('_line_break_flags', (None, re.X))
    def test_splitter_doesnt_hang(self, _sep, _sep_flags,
        _line_break, _line_break_flags
    ):

        # skip impossible -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        if isinstance(_sep, re.Pattern) and _sep_flags:
            pytest.skip(reason=f'cant have re.compile and flags')
        if isinstance(_line_break, re.Pattern) and _line_break_flags:
            pytest.skip(reason=f'cant have re.compile and flags')
        # END skip impossible -- -- -- -- -- -- -- -- -- -- -- -- -- --


        _X = ['sknaspdouralmnbpasoiruaaskdrua']

        assert isinstance(
            _splitter(_X, _sep, _sep_flags, _line_break, _line_break_flags),
            list
        )



    def test_accuracy(self):

        ex1 = [
            'want to split. on a period here.',
            'no-op here',
            'want to split, on comma here',
            'zebras split on z and q here',
            'they split at the end in alcatraz',
            'last split on; a semicolon.',
        ]

        # it is important that in ex1 q is after z but in _sep z is after q
        _sep = '[;QZ]'
        _line_break = re.compile('[.,]')

        out = _splitter(ex1, _sep, re.I, _line_break, None)

        assert isinstance(out, list)
        assert all(map(isinstance, out, (str for _ in out)))

        exp = [
            'want to split.',
            ' on a period here.',
            'no-op here',
            'want to split,',
            ' on comma here',
            'z',
            'ebras split on z',
            ' and q',
            ' here',
            'they split at the end in alcatraz',
            'last split on;',
            ' a semicolon.'
        ]

        for _idx in range(len(exp)):
            assert np.array_equal(out[_idx], exp[_idx])








