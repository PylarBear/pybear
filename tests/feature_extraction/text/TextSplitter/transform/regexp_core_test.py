# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import re

import numpy as np

from pybear.feature_extraction.text._TextSplitter._transform._regexp_core \
    import _regexp_core



class TestStrCore:



    @staticmethod
    @pytest.fixture(scope='module')
    def _text():
        return [
            "Double, double toil and trouble;",
            "Fire burn, and cauldron bubble.",
            "Fillet of a fenny snake",
            "In the cauldron boil and bake.",
            "Eye of newt and toe of frog,",
            "Wool of bat and tongue of dog,",
            "Adder’s fork and blindworm’s sting,",
            "Lizard’s leg and howlet’s wing,",
            "For a charm of powerful trouble,",
            "Like a hell-broth boil and bubble.",
            "Double, double toil and trouble;",
            "Fire burn, and cauldron bubble."
        ]



    # no validation


    def test_accuracy(self, _text):

        # sep is regexp
        # sep is re.compile
        # sep is list[False | regexp | re.compile]

        # sep is str -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        # maxsplit is default (all)
        out = _regexp_core(_text[:2], '[,]', None, None)
        assert isinstance(out, list)
        for __ in out:
            assert all(map(isinstance, __, (str for _ in __)))

        assert all(map(np.array_equal, out,
            [["Double", " double toil and trouble;"],
            ["Fire burn", " and cauldron bubble."]]
        ))


        # maxsplit is -1
        out = _regexp_core(_text[:2], '[,]', -1, None)
        assert isinstance(out, list)
        for __ in out:
            assert all(map(isinstance, __, (str for _ in __)))

        assert all(map(np.array_equal, out,
            [["Double, double toil and trouble;"],
            ["Fire burn, and cauldron bubble."]]
        ))

        # flags is re.I
        out = _regexp_core(_text[:2], '[d]', None, re.I)
        assert isinstance(out, list)
        for __ in out:
            assert all(map(isinstance, __, (str for _ in __)))

        assert all(map(np.array_equal, out,
            [["", "ouble, ", "ouble toil an", " trouble;"],
            ["Fire burn, an", " caul", "ron bubble."]]
        ))

        # END sep is str -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # sep is re.compile -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        # maxsplit is default (all)
        out = _regexp_core(_text[:2], re.compile('[, ]'), None, None)
        assert isinstance(out, list)
        for __ in out:
            assert all(map(isinstance, __, (str for _ in __)))

        assert all(map(np.array_equal, out,
            [["Double", "", "double", "toil", "and", "trouble;"],
            ["Fire", "burn", "", "and", "cauldron", "bubble."]]
        ))


        # maxsplit is 2
        out = _regexp_core(_text[:2], re.compile('[\s,]'), 2, None)
        assert isinstance(out, list)
        for __ in out:
            assert all(map(isinstance, __, (str for _ in __)))

        assert all(map(np.array_equal, out,
            [["Double", "", "double toil and trouble;"],
            ["Fire", "burn", " and cauldron bubble."]],
        ))

        # flags is re.I
        with pytest.raises(ValueError):
            # ValueError: cannot process flags argument with a compiled pattern
            out = _regexp_core(_text[:2], re.compile('[d]'), None, re.I)
        # assert isinstance(out, list)
        # for __ in out:
        #     assert all(map(isinstance, __, (str for _ in __)))
        #
        # assert all(map(np.array_equal, out,
        #     [["", "ouble, ", "ouble toil and trouble;"],
        #     ["Fire burn, and caul", "ron bubble."]],
        # ))
        # END sep is re.compile -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # sep is list -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        _seps = [False, False, '[\s]', re.compile('[bl]')]

        # maxsplit is default (all)

        out = _regexp_core(_text[:4], _seps, None, None)
        assert isinstance(out, list)
        for __ in out:
            assert all(map(isinstance, __, (str for _ in __)))

        assert all(map(np.array_equal, out,
            [["Double, double toil and trouble;"],
            ["Fire burn, and cauldron bubble."],
            ["Fillet", "of", "a", "fenny", "snake"],
            ["In the cau", "dron ", "oi", " and ", "ake."]]
        ))


        # maxsplit is 2
        out = _regexp_core(_text[:4], _seps, 2, None)
        assert isinstance(out, list)
        for __ in out:
            assert all(map(isinstance, __, (str for _ in __)))

        assert all(map(np.array_equal, out,
            [["Double, double toil and trouble;"],
            ["Fire burn, and cauldron bubble."],
            ["Fillet", "of", "a fenny snake"],
            ["In the cau", "dron ", "oil and bake."]]
        ))

        # flags is re.I
        with pytest.raises(ValueError):
            # ValueError: cannot process flags argument with a compiled pattern
            out = _regexp_core(_text[:4], _seps, None, re.I)
        # assert isinstance(out, list)
        # for __ in out:
        #     assert all(map(isinstance, __, (str for _ in __)))
        #
        # assert all(map(np.array_equal, out,
        #     [["Double, double toil and trouble;"],
        #     ["Fire burn, and cauldron bubble."],
        #     ["Fillet", "of", "a", "fenny", "snake"],
        #     ["In the cau", "dron ", "oi", " and ", "ake."]]
        # ))

        # END sep is list -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    def test_falses_spread_around_correctly(self, _text):

        # _seps is default

        # maxsplit is list -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        _maxsplit = [2, 2, False, False]

        out = _regexp_core(_text[:4], '[\s]', _maxsplit, None)
        assert isinstance(out, list)
        for __ in out:
            assert all(map(isinstance, __, (str for _ in __)))

        assert all(map(np.array_equal, out,
            [["Double,", "double", "toil and trouble;"],
            ["Fire", "burn,", "and cauldron bubble."],
            ["Fillet of a fenny snake"],
            ["In the cauldron boil and bake."]]
        ))


        # flags is re.I
        out = _regexp_core(_text[:4], '[f]', _maxsplit, re.I)
        assert isinstance(out, list)
        for __ in out:
            assert all(map(isinstance, __, (str for _ in __)))

        assert all(map(np.array_equal, out,
            [["Double, double toil and trouble;"],
            ["", "ire burn, and cauldron bubble."],
            ["Fillet of a fenny snake"],
            ["In the cauldron boil and bake."]]
        ))

        # END maxsplit is list -- -- -- -- -- -- -- -- -- -- -- -- -- --


        # flags is list -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        _flags = [re.I, False, None, re.I]

        out = _regexp_core(_text[:4], '[d]', None, _flags)
        assert isinstance(out, list)
        for __ in out:
            assert all(map(isinstance, __, (str for _ in __)))

        assert all(map(np.array_equal, out,
            [["", "ouble, ", "ouble toil an", " trouble;"],
            ["Fire burn, and cauldron bubble."],
            ["Fillet of a fenny snake"],
            ["In the caul", "ron boil an", " bake."]]
        ))


        # maxsplit is 1
        out = _regexp_core(_text[:4], '[d]', 1, _flags)
        assert isinstance(out, list)
        for __ in out:
            assert all(map(isinstance, __, (str for _ in __)))

        assert all(map(np.array_equal, out,
            [["", "ouble, double toil and trouble;"],
            ["Fire burn, and cauldron bubble."],
            ["Fillet of a fenny snake"],
            ["In the caul", "ron boil and bake."]]
        ))

        # END flags is list -- -- -- -- -- -- -- -- -- -- -- -- -- --





