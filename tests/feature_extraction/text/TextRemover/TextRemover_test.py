# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np
import pandas as pd
import polars as pl

from pybear.feature_extraction.text._TextRemover.TextRemover import TextRemover




class TestTextRemover:


    @staticmethod
    @pytest.fixture(scope='module')
    def _words():
        return list('abcde')



    def test_empty_X(self):

        # 1D
        TestCls = TextRemover(str_remove=',')

        out = TestCls.transform([])

        assert isinstance(out, list)
        assert len(out) == 0

        # 2D -- returns empty 1D
        TestCls = TextRemover(regexp_remove='[n-z]')

        out = TestCls.transform([[]])

        assert isinstance(out, list)
        assert len(out) == 0


    def test_str_remove_1(self, _words):

        TestCls = TextRemover(str_remove={'a', "c"})

        out = TestCls.transform(_words, copy=True)
        assert isinstance(out, list)
        assert all(map(isinstance, out, (str for _ in out)))

        assert np.array_equal(out, list('bde'))


    def test_str_remove_2(self, _words):

        TestCls = TextRemover(str_remove='a')

        out = TestCls.transform(list(map(list, _words)), copy=True)
        assert isinstance(out, list)
        for _ in out:
            assert isinstance(_, list)
            assert all(map(isinstance, _, (str for i in _)))

        assert all(map(
            np.array_equal,
            out,
            [['b'], ['c'], ['d'], ['e']]
        ))


    def test_re_split_1(self, _words):

        TestCls = TextRemover(regexp_remove="[a-c]+")

        out = TestCls.transform(_words, copy=True)
        assert isinstance(out, list)
        assert all(map(isinstance, out, (str for i in out)))

        assert np.array_equal(out, list('de'))


    def test_re_split_2(self, _words):

        TestCls = TextRemover(regexp_remove=[False, ".", False, False, False])

        out = TestCls.transform(list(map(list, _words)), copy=True)
        assert isinstance(out, list)
        for _ in out:
            assert isinstance(_, list)
            assert all(map(isinstance, _, (str for i in _)))

        assert all(map(
            np.array_equal,
            out,
            [['a'], ['c'], ['d'], ['e']]
        ))


    def test_various_input_containers(self, _words):

        TestCls = TextRemover(str_remove="e")


        # python list accepted
        out = TestCls.transform(_words, copy=True)
        assert isinstance(out, list)
        assert np.array_equal(out, list('abcd'))

        # python 1D tuple accepted
        out = TestCls.transform(tuple(_words), copy=True)
        assert isinstance(out, list)
        assert np.array_equal(out, list('abcd'))

        # python 1D set accepted
        out = TestCls.transform(set(_words), copy=True)
        assert isinstance(out, list)
        assert np.array_equal(sorted(out), sorted(list('abcd')))

        # np 1D accepted
        out = TestCls.transform(np.array(_words), copy=True)
        assert isinstance(out, list)
        assert np.array_equal(out, list('abcd'))

        # pd series accepted
        out = TestCls.transform(pd.Series(_words), copy=True)
        assert isinstance(out, list)
        assert np.array_equal(out, list('abcd'))

        # polars series accepted
        out = TestCls.transform(pl.Series(_words), copy=True)
        assert isinstance(out, list)
        assert np.array_equal(out, list('abcd'))

        # np 2D accepted
        out = TestCls.transform(
            np.array(_words).reshape((len(_words), -1)),
            copy=True
        )
        assert isinstance(out, list)
        assert np.array_equal(out, [['a'], ['b'], ['c'], ['d']])

        # pd DataFrame accepted
        TestCls.transform(
            pd.DataFrame(np.array(_words).reshape((len(_words), -1))),
            copy=True
        )
        assert isinstance(out, list)
        assert np.array_equal(out, [['a'], ['b'], ['c'], ['d']])

        # polars 2D accepted
        out = TestCls.transform(
            pl.from_numpy(np.array(_words).reshape((len(_words), -1))),
            copy=True
        )
        assert isinstance(out, list)
        assert np.array_equal(out, [['a'], ['b'], ['c'], ['d']])








