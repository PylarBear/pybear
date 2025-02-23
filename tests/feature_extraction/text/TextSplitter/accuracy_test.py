# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.feature_extraction.text._TextSplitter.TextSplitter import TextSplitter




class TestAccuracy:


    @staticmethod
    @pytest.fixture(scope='module')
    def _words():
        return [
            "Scale of dragon, tooth of wolf",
            "Witch’s mummy, maw and gulf"
        ]



    def test_empty_X(self):

        TestCls = TextSplitter()

        out = TestCls.transform([])

        assert isinstance(out, list)
        assert len(out) == 0


    def test_str_split_1(self, _words):

        TestCls = TextSplitter(str_sep={',', "’", ' '})

        out = TestCls.transform(_words, copy=True)
        assert isinstance(out, list)
        for _ in out:
            assert isinstance(_, list)
            assert all(map(isinstance, _, (str for i in _)))

        assert all(map(
            np.array_equal,
            out,
            [
                ["Scale", "of", "dragon", "", "tooth", "of", "wolf"],
                ["Witch", "s", "mummy", "", "maw", "and", "gulf"]
            ]
            ))


    def test_str_split_2(self, _words):

        TestCls = TextSplitter(str_sep=[{',', "’", ' '}, False])

        out = TestCls.transform(_words, copy=True)
        assert isinstance(out, list)
        for _ in out:
            assert isinstance(_, list)
            assert all(map(isinstance, _, (str for i in _)))

        assert all(map(
            np.array_equal,
            out,
            [
                ["Scale", "of", "dragon", "", "tooth", "of", "wolf"],
                ["Witch’s mummy, maw and gulf"]
            ]
            ))


    def test_re_split_1(self, _words):

        TestCls = TextSplitter(regexp_sep="[\s’,]")

        out = TestCls.transform(_words, copy=True)
        assert isinstance(out, list)
        for _ in out:
            assert isinstance(_, list)
            assert all(map(isinstance, _, (str for i in _)))

        assert all(map(
            np.array_equal,
            out,
            [
                ["Scale", "of", "dragon", "", "tooth", "of", "wolf"],
                ["Witch", "s", "mummy", "", "maw", "and", "gulf"]
            ]
            ))


    def test_re_split_2(self, _words):

        TestCls = TextSplitter(regexp_sep=["[\s’,]", False])

        out = TestCls.transform(_words, copy=True)
        assert isinstance(out, list)
        for _ in out:
            assert isinstance(_, list)
            assert all(map(isinstance, _, (str for i in _)))

        assert all(map(
            np.array_equal,
            out,
            [
                ["Scale", "of", "dragon", "", "tooth", "of", "wolf"],
                ["Witch’s mummy, maw and gulf"]
            ]
            ))








