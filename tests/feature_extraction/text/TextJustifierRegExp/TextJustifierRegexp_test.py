# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



# takes any y
# fit_transform
# set_params
# get_params
# transform data is longer than fitted data
# test accepts 1D & 2D array-like


import pytest

import numpy as np
import pandas as pd
import polars as pl

from pybear.feature_extraction.text._TextJustifierRegExp.TextJustifierRegExp import \
    TextJustifierRegExp as TJRE



class TestTextJustifierRegExp:


    @staticmethod
    @pytest.fixture(scope='function')
    def _kwargs():
        return {
            'n_chars': 20,
            'sep': '\ ',
            'sep_flags': None,
            'line_break': '\.',
            'line_break_flags': None,
            'backfill_sep': ' ',
            'join_2D': ' '
        }


    @staticmethod
    @pytest.fixture(scope='function')
    def _text():
        return [
            "Round about the cauldron go;",
            "In the poisoned entrails throw.",
            "Toad, that under cold stone",
            "Days and nights has thirty-one",
            "Sweltered venom sleeping got,",
            "Boil thou first i’ th’ charmèd pot."
        ]


    @pytest.mark.parametrize('y', ([1,2], None, {1,2}, 'junk'))
    def test_takes_any_y(self, _kwargs, _text, y):

        TestCls = TJRE(**_kwargs)

        TestCls.partial_fit(_text, y)

        TestCls.fit(_text, y)

        TestCls.fit_transform(_text, y)

        TestCls.score(_text, y)


    @pytest.mark.parametrize('deep', (True, False))
    def test_get_params(self, _kwargs, deep):

        TestCls = TJRE(**_kwargs)

        out = TestCls.get_params(deep)

        assert isinstance(out, dict)
        assert 'sep' in out
        assert out['sep'] == '\\ '


    def test_set_params(self, _kwargs):

        TestCls = TJRE(**_kwargs)

        assert isinstance(TestCls.set_params(**{'sep': ','}), TJRE)

        assert TestCls.sep == ','

        out = TestCls.get_params()

        assert isinstance(out, dict)
        assert 'sep' in out
        assert out['sep'] == ','


    def test_accuracy(self, _kwargs, _text):

        TestCls = TJRE(**_kwargs)

        out = TestCls.transform(_text, copy=True)
        assert isinstance(out, list)
        assert all(map(isinstance, out, (str for _ in out)))

        exp = [
            "Round about the ",
            "cauldron go; In the ",
            "poisoned entrails ",
            "throw.",
            "Toad, that under ",
            "cold stone Days and ",
            "nights has ",
            "thirty-one",
            "Sweltered venom ",
            "sleeping got, Boil ",
            "thou first i’ th’ ",
            "charmèd pot."
        ]

        assert np.array_equal(out, exp)


    def test_various_1D_input_containers(self, _kwargs):

        _base_text = [
            "Fillet of a fenny snake",
            "In the cauldron boil and bake.",
            "Eye of newt and toe of frog,"
        ]

        _exp = [
            "Fillet of a fenny ",
            "snake In the ",
            "cauldron boil and ",
            "bake.",
            "Eye of newt and toe ",
            "of frog,"
        ]

        TestCls = TJRE(**_kwargs)


        # python list accepted
        out = TestCls.transform(list(_base_text))
        assert isinstance(out, list)
        assert all(map(isinstance, out, (str for _ in out)))
        assert np.array_equal(out, _exp)

        # python 1D tuple accepted
        out = TestCls.transform(tuple(_base_text))
        assert isinstance(out, list)
        assert all(map(isinstance, out, (str for _ in out)))
        assert np.array_equal(out, _exp)

        # python 1D set accepted
        out = TestCls.transform(set(_base_text))
        assert isinstance(out, list)
        assert all(map(isinstance, out, (str for _ in out)))
        # dont bother checking for accuracy

        # np 1D accepted
        out = TestCls.transform(np.array(_base_text))
        assert isinstance(out, list)
        assert all(map(isinstance, out, (str for _ in out)))
        assert np.array_equal(out, _exp)

        # pd series accepted
        out = TestCls.transform(pd.Series(_base_text))
        assert isinstance(out, list)
        assert all(map(isinstance, out, (str for _ in out)))
        assert np.array_equal(out, _exp)

        # polars series accepted
        out = TestCls.transform(pl.Series(_base_text))
        assert isinstance(out, list)
        assert all(map(isinstance, out, (str for _ in out)))
        assert np.array_equal(out, _exp)


    def test_various_2D_input_containers(self, _kwargs):

        _base_text = [
            ["Fillet", "of", "a", "fenny", "snake"],
            ["In", "the", "cauldron", "boil", "and"],
            ["Eye", "of", "newt", "and", "toe"]
        ]

        _exp = [
            ["Fillet", "of", "a", "fenny"],
            ["snake", "In", "the"],
            ["cauldron", "boil", "and"],
            ["Eye", "of", "newt", "and", "toe"]
        ]

        TestCls = TJRE(**_kwargs)


        # python 2D list accepted
        out = TestCls.transform(list(map(list, _base_text)))
        assert isinstance(out, list)
        print(f'pizza print {out=}')
        for r_idx in range(len(out)):
            assert isinstance(out[r_idx], list)
            assert all(map(isinstance, out[r_idx], (str for _ in out[r_idx])))
            assert np.array_equal(out[r_idx], _exp[r_idx])

        # python 2D tuple accepted
        out = TestCls.transform(tuple(map(tuple, _base_text)))
        assert isinstance(out, list)
        for r_idx in range(len(out)):
            assert isinstance(out[r_idx], list)
            assert all(map(isinstance, out[r_idx], (str for _ in out[r_idx])))
            assert np.array_equal(out[r_idx], _exp[r_idx])

        # np 2D accepted
        out = TestCls.transform(np.array(_base_text))
        assert isinstance(out, list)
        for r_idx in range(len(out)):
            assert isinstance(out[r_idx], list)
            assert all(map(isinstance, out[r_idx], (str for _ in out[r_idx])))
            assert np.array_equal(out[r_idx], _exp[r_idx])

        # pd DataFrame accepted
        out = TestCls.transform(
            pd.DataFrame(np.array(_base_text))
        )
        assert isinstance(out, list)
        for r_idx in range(len(out)):
            assert isinstance(out[r_idx], list)
            assert all(map(isinstance, out[r_idx], (str for _ in out[r_idx])))
            assert np.array_equal(out[r_idx], _exp[r_idx])

        # polars 2D accepted
        out = TestCls.transform(
            pl.from_numpy(np.array(_base_text))
        )
        assert isinstance(out, list)
        for r_idx in range(len(out)):
            assert isinstance(out[r_idx], list)
            assert all(map(isinstance, out[r_idx], (str for _ in out[r_idx])))
            assert np.array_equal(out[r_idx], _exp[r_idx])












