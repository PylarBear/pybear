# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np
import pandas as pd
import polars as pl

from pybear.preprocessing._NanStandardizer.NanStandardizer import \
    NanStandardizer

from pybear.utilities._nan_masking import nan_mask



class TestAccuracy:


    @staticmethod
    @pytest.fixture(scope='module')
    def _X():
        __ = np.random.uniform(0, 1, (5,3))
        __[0, 1] = np.nan
        __[1, 0] = np.nan
        __[4, 2] = np.nan
        return __



    def test_empty_X(self):

        # 1D
        TestCls = NanStandardizer()

        out = TestCls.transform(np.zeros((0,)))

        assert isinstance(out, np.ndarray)
        assert len(out) == 0

        # 2D
        TestCls = NanStandardizer()

        out = TestCls.transform(np.zeros((1,0)))

        assert isinstance(out, np.ndarray)
        assert len(out) == 1
        assert len(out[0]) == 0


    def test_accuracy(self, _X):

        # default fill = np.nan

        TestCls = NanStandardizer()

        out = TestCls.transform(_X, copy=True)
        assert isinstance(out, np.ndarray)
        assert all(map(isinstance, out, (np.ndarray for _ in out)))

        ref = _X.copy()
        ref[nan_mask(ref)] = np.nan
        assert np.array_equal(out, ref, equal_nan=True)


    def test_various_input_containers(self, _X):

        TestCls = NanStandardizer()

        # python list rejected
        with pytest.raises(TypeError):
            TestCls.transform(list(_X[:, 0]), copy=True)

        # python 1D tuple rejected
        with pytest.raises(TypeError):
            TestCls.transform(tuple(_X[:, 0]), copy=True)

        # python 1D set rejected
        with pytest.raises(TypeError):
            TestCls.transform(set(_X[:, 0]), copy=True)

        # np 1D accepted
        out = TestCls.transform(np.array(_X[:, 0]), copy=True)
        assert isinstance(out, np.ndarray)
        assert np.array_equal(out, _X[:, 0], equal_nan=True)

        # pd series accepted
        out = TestCls.transform(pd.Series(_X[:, 0]), copy=True)
        assert isinstance(out, pd.core.series.Series)
        assert np.array_equal(out, _X[:, 0], equal_nan=True)

        # polars series accepted
        out = TestCls.transform(pl.Series(_X[:, 0]), copy=True)
        assert isinstance(out, pl.Series)
        assert np.array_equal(out, _X[:, 0], equal_nan=True)

        # np 2D accepted
        out = TestCls.transform(_X, copy=True)
        assert isinstance(out, np.ndarray)
        assert np.array_equal(out, _X, equal_nan=True)

        # pd DataFrame accepted
        out = TestCls.transform(pd.DataFrame(_X, copy=True))
        assert isinstance(out, pd.core.frame.DataFrame)
        assert np.array_equal(out.to_numpy(), _X, equal_nan=True)

        # polars 2D accepted
        out = TestCls.transform(pl.DataFrame(_X), copy=True)

        assert isinstance(out, pl.DataFrame)
        assert all(map(
            np.array_equal,
            out.to_numpy(),
            _X,
            (True for _ in out)
        ))








