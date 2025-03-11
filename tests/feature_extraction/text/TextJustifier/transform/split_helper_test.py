# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.feature_extraction.text._TextJustifier._transform._split_helper \
    import _split_helper




class TestSplitHelper:

    # no validation

    def test_accuracy(self):

        _str = 'this will be split on idx 7'
        _idx = 7

        out = _split_helper(_str, _idx)

        assert isinstance(out, tuple)
        assert len(out) == 2
        assert isinstance(out[0], str)
        assert isinstance(out[1], str)
        assert out[0] == 'this wil'
        assert out[1] == 'l be split on idx 7'

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        _str = 'this will be split on idx 0'
        _idx = 0

        out = _split_helper(_str, _idx)

        assert isinstance(out, tuple)
        assert len(out) == 2
        assert isinstance(out[0], str)
        assert isinstance(out[1], str)
        assert out[0] == 't'
        assert out[1] == 'his will be split on idx 0'

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        _str = 'this will be split on the last idx'
        _idx = len(_str) - 1

        out = _split_helper(_str, _idx)

        assert isinstance(out, tuple)
        assert len(out) == 2
        assert isinstance(out[0], str)
        assert isinstance(out[1], str)
        assert out[0] == 'this will be split on the last idx'
        assert out[1] == ''

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


        _str = 'this will be split outside'
        _idx = 100

        out = _split_helper(_str, _idx)

        assert isinstance(out, tuple)
        assert len(out) == 2
        assert isinstance(out[0], str)
        assert isinstance(out[1], str)
        assert out[0] == 'this will be split outside'
        assert out[1] == ''

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


