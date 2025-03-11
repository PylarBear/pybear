# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.feature_extraction.text._TextJustifier._transform._sep_splitter \
    import _sep_splitter



class TestLinebreakSplitter:

    # no validation

    def test_accuracy(self):

        ex1 = [
            'want to split. on a period here.',
            'no-op here',
            'want to split, on comma here',
            'zebras split on z here',
            'last split on; a semicolon.',
        ]

        _sep = {'.', ',', ';', 'z'}

        exp = [
            'want to split.',
            ' on a period here.',
            'no-op here',
            'want to split,',
            ' on comma here',
            'z',
            'ebras split on z',
            ' here',
            'last split on;',
            ' a semicolon.'
        ]

        out = _sep_splitter(ex1, _sep)

        assert isinstance(out, list)
        assert all(map(isinstance, out, (str for _ in out)))

        for _idx in range(len(exp)):
            assert np.array_equal(out[_idx], exp[_idx])








