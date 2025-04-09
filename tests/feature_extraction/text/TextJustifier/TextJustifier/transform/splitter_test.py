# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.feature_extraction.text._TextJustifier._TextJustifier._transform. \
    _splitter import _splitter



class TestSplitter:

    # no validation

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
        _sep = {';', 'q', 'z'}
        _line_break = {'.', ','}

        out = _splitter(ex1, _sep, _line_break)

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








