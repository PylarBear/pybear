# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numbers

import pytest

import numpy as np

from pybear.feature_extraction.text._TextPadder._partial_fit._partial_fit import \
    _partial_fit



class TestPartialFit:

    # no validation

    def test_accuracy(self):

        out = _partial_fit(np.random.randint(0, 10, (37, 13)))

        assert isinstance(out, numbers.Integral)
        assert out == 13

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        out = _partial_fit(
            [
                list('abc'),
                list('abcdef'),
                list('abcde')
            ]
        )

        assert isinstance(out, numbers.Integral)
        assert out == 6

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        out = _partial_fit(
            (
                tuple(list('abcdefghijkl')),
                tuple(list('abcdefgh')),
                tuple(list('abcdefghij'))
            )
        )

        assert isinstance(out, numbers.Integral)
        assert out == 12

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        out = _partial_fit(
            [
                ['double,', 'double', 'toil', 'and', 'trouble.'],
                ['In', 'the', 'cauldron', 'boil', 'and', 'bubble']
            ]
        )

        assert isinstance(out, numbers.Integral)
        assert out == 6

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --







