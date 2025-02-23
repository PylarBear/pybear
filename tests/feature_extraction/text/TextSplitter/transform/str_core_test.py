# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.feature_extraction.text._TextSplitter._transform._str_core import \
    _str_core



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

        # sep is None
        # sep is str
        # sep is set[str]
        # sep is list

        # sep is None -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        # maxsplit is default (all)
        out = _str_core(_text[:2], None, None)
        assert isinstance(out, list)
        for __ in out:
            assert all(map(isinstance, __, (str for _ in __)))

        assert all(map(np.array_equal, out,
            [['Double,', 'double', 'toil', 'and', 'trouble;'],
             ['Fire', 'burn,', 'and', 'cauldron', 'bubble.']]
        ))


        # maxsplit is 2
        out = _str_core(_text[:2], None, 2)
        assert isinstance(out, list)
        for __ in out:
            assert all(map(isinstance, __, (str for _ in __)))

        assert all(map(np.array_equal, out,
            [['Double,', 'double', 'toil and trouble;'],
            ['Fire', 'burn,', 'and cauldron bubble.']]
        ))
        # END sep is None -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # sep is str -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        # maxsplit is default (all)
        out = _str_core(_text[:2], ',', None)
        assert isinstance(out, list)
        for __ in out:
            assert all(map(isinstance, __, (str for _ in __)))

        assert all(map(np.array_equal, out,
            [["Double", " double toil and trouble;"],
            ["Fire burn", " and cauldron bubble."]]
        ))


        # maxsplit is 0
        out = _str_core(_text[:2], ',', 0)
        assert isinstance(out, list)
        for __ in out:
            assert all(map(isinstance, __, (str for _ in __)))

        assert all(map(np.array_equal, out,
            [["Double, double toil and trouble;"],
            ["Fire burn, and cauldron bubble."]]
        ))
        # END sep is str -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # sep is set -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        # maxsplit is default (all)
        out = _str_core(_text[:2], {',', ' '}, None)
        assert isinstance(out, list)
        for __ in out:
            assert all(map(isinstance, __, (str for _ in __)))

        assert all(map(np.array_equal, out,
            [["Double", "", "double", "toil", "and", "trouble;"],
            ["Fire", "burn", "", "and", "cauldron", "bubble."]]
        ))


        # maxsplit is 2
        out = _str_core(_text[:2], {',', ' '}, 2)
        assert isinstance(out, list)
        for __ in out:
            assert all(map(isinstance, __, (str for _ in __)))

        assert all(map(np.array_equal, out,
            [["Double", "", "double toil and trouble;"],
            ["Fire", "burn", " and cauldron bubble."]],
        ))
        # END sep is set -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # sep is list -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        _seps = [False, False, ' ', {'b', 'l'}]

        # maxsplit is default (all)

        out = _str_core(_text[:4], _seps, None)
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
        out = _str_core(_text[:4], _seps, 2)
        assert isinstance(out, list)
        for __ in out:
            assert all(map(isinstance, __, (str for _ in __)))

        assert all(map(np.array_equal, out,
            [["Double, double toil and trouble;"],
            ["Fire burn, and cauldron bubble."],
            ["Fillet", "of", "a fenny snake"],
            ["In the cau", "dron ", "oil and bake."]]
        ))
        # END sep is list -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    def test_falses_spread_around_correctly(self, _text):

        # _seps is default

        # maxsplit is list -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        _maxsplit = [2, 2, False, False]

        out = _str_core(_text[:4], None, _maxsplit)
        assert isinstance(out, list)
        for __ in out:
            assert all(map(isinstance, __, (str for _ in __)))

        assert all(map(np.array_equal, out,
            [["Double,", "double", "toil and trouble;"],
            ["Fire", "burn,", "and cauldron bubble."],
            ["Fillet of a fenny snake"],
            ["In the cauldron boil and bake."]]
        ))

        # END maxsplit is list -- -- -- -- -- -- -- -- -- -- -- -- -- --


