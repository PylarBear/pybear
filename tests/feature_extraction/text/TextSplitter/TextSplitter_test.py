# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np
import pandas as pd
import polars as pl

from pybear.feature_extraction.text._TextSplitter.TextSplitter import TextSplitter




class TestTextSplitter:


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


    def test_various_input_containers(self):

        _base_text = [
            "Scale of dragon",
            "Witch’s mummy"
        ]

        _exp = [
            ["Scale", "of", "dragon"],
            ["Witch’s", "mummy"]
        ]

        TestCls = TextSplitter(str_sep=" ")


        # python list accepted
        out = TestCls.transform(list(_base_text))
        assert isinstance(out, list)
        assert all(map(np.array_equal, out, _exp))

        # python 1D tuple accepted
        out = TestCls.transform(tuple(_base_text))
        assert isinstance(out, list)
        assert all(map(np.array_equal, out, _exp))

        # python 1D set accepted
        out = TestCls.transform(set(_base_text))
        assert isinstance(out, list)
        assert all(map(np.array_equal, sorted(out), sorted(_exp)))

        # np 1D accepted
        out = TestCls.transform(np.array(_base_text))
        assert isinstance(out, list)
        assert all(map(np.array_equal, out, _exp))

        # pd series accepted
        out = TestCls.transform(pd.Series(_base_text))
        assert isinstance(out, list)
        assert all(map(np.array_equal, out, _exp))

        # polars series accepted
        out = TestCls.transform(pl.Series(_base_text))
        assert isinstance(out, list)
        assert all(map(np.array_equal, out, _exp))

        # np 2D rejected
        with pytest.raises(TypeError):
            TestCls.transform(np.array(_base_text).reshape((2, -1)))

        # pd DataFrame rejected
        with pytest.raises(TypeError):
            TestCls.transform(
                pd.DataFrame(np.array(_base_text).reshape((2, -1)))
            )

        # polars 2D rejected
        with pytest.raises(TypeError):
            TestCls.transform(
                pl.DataFrame(np.array(_base_text).reshape((2, -1)))
            )






