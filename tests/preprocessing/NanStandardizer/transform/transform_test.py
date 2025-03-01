# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as ss

from pybear.preprocessing._NanStandardizer._transform._transform import _transform

from pybear.utilities._nan_masking import nan_mask



class TestTransform:


    @staticmethod
    @pytest.fixture(scope='function')
    def _X_num():

        _shape = (5,3)

        while True:
            # make sure all columns get at least one nan
            __ = np.random.randint(0, 3, _shape).astype(np.float64)
            for _ in range(5):
                _rand_r_idx = np.random.choice(_shape[0])
                _rand_c_idx = np.random.choice(_shape[1])
                __[_rand_r_idx, _rand_c_idx] = np.nan

            for c_idx in range(_shape[1]):
                if not any(nan_mask(__[:, c_idx])):
                    break
            else:
                return __


    @staticmethod
    @pytest.fixture(scope='function')
    def _X_str():

        _shape = (5,3)

        while True:
            # make sure all columns get at least one nan
            __ = np.random.choice(list('abcde'), _shape, replace=True).astype('<U3')
            for _ in range(5):
                _rand_r_idx = np.random.choice(_shape[0])
                _rand_c_idx = np.random.choice(_shape[1])
                __[_rand_r_idx, _rand_c_idx] = 'nan'

            for c_idx in range(_shape[1]):
                if not any(nan_mask(__[:, c_idx])):
                    break
            else:
                return __

    # END fixtures v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^


    @pytest.mark.parametrize('_dtype', ('num', 'str'))
    @pytest.mark.parametrize('_fill', (99, None, True, 'NaN'))
    def test_accuracy_np(self, _fill, _dtype, _X_num, _X_str):


        if _dtype == 'num':
            _X = _X_num
        elif _dtype == 'str':
            _X = _X_str
        else:
            raise Exception

        out = _transform(_X, _fill)

        assert isinstance(out, type(_X))
        assert out.shape == _X.shape

        ref = _X.copy()
        ref[nan_mask(ref)] = _fill

        # assertions -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        if (isinstance(_fill , str) or _dtype == 'str'):
            # fill was a str, which should coerce target object to str dtype
            # or target already was str dtype
            if len(out.shape) == 1:
                assert np.array_equal(
                    list(map(str, out)),
                    list(map(str, ref))
                )
            else:
                for r_idx in range(out.shape[0]):
                    assert np.array_equal(
                        list(map(str, out[r_idx])),
                        list(map(str, ref[r_idx]))
                    )
        else:
            assert all(map(np.array_equal, out, ref, (True for _ in out)))


    @pytest.mark.parametrize('X_format', ('pd_series', 'pd_df'))
    @pytest.mark.parametrize('_dtype', ('num', 'str'))
    @pytest.mark.parametrize('_fill', (99, None, True, 'NaN'))
    def test_accuracy_pd(self, X_format, _fill, _dtype, _X_num, _X_str):


        if _dtype == 'num':
            _X = _X_num
        elif _dtype == 'str':
            _X = _X_str
        else:
            raise Exception

        if X_format == 'pd_series':
            _X = pd.Series(_X[:, 0])
        elif X_format == 'pd_df':
            _X = pd.DataFrame(_X)
        else:
            raise Exception


        out = _transform(_X, _fill)

        assert isinstance(out, type(_X))
        assert out.shape == _X.shape

        ref = _X.copy()
        ref[nan_mask(ref)] = _fill

        # assertions -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        if (isinstance(_fill , str) or _dtype == 'str'):
            # fill was a str, which should coerce target object to str dtype
            # or target already was str dtype
            if len(out.shape) == 1:
                assert np.array_equal(
                    list(map(str, out)),
                    list(map(str, ref))
                )
            else:
                for r_idx in range(len(out)):
                    assert np.array_equal(
                        list(map(str, out.iloc[r_idx, :])),
                        list(map(str, ref.iloc[r_idx, :]))
                    )
        else:
            assert all(map(np.array_equal, out, ref, (True for _ in out)))


    @pytest.mark.parametrize('X_format',
        ('ss_csr_mat', 'ss_csr_arr', 'ss_csc_mat', 'ss_csc_arr', 'ss_coo_mat',
        'ss_coo_arr')
    )
    @pytest.mark.parametrize('_fill', (99, None, True, np.nan))
    def test_accuracy_ss(self, X_format, _fill, _X_num):

        # no strings!

        _X = _X_num

        if X_format == 'ss_csr_mat':
            _X = ss.csr_matrix(_X)
        elif X_format == 'ss_csr_arr':
            _X = ss.csr_array(_X)
        elif X_format == 'ss_csc_mat':
            _X = ss.csc_matrix(_X)
        elif X_format == 'ss_csc_arr':
            _X = ss.csc_array(_X)
        elif X_format == 'ss_coo_mat':
            _X = ss.coo_matrix(_X)
        elif X_format == 'ss_coo_arr':
            _X = ss.coo_array(_X)
        else:
            raise Exception


        out = _transform(_X, _fill)

        assert isinstance(out, type(_X))
        assert out.shape == _X.shape


        ref = _X.data.copy()
        ref[nan_mask(ref)] = _fill

        # assertions -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        assert np.array_equal(out.data, ref.data, equal_nan=True)


    @pytest.mark.parametrize('X_format', ('pl_series', 'pl_df'))
    @pytest.mark.parametrize('_dtype', ('num', 'str'))
    @pytest.mark.parametrize('_fill', (99, None, True, 'NaN'))
    def test_accuracy_pl(self, X_format, _fill, _dtype, _X_num, _X_str):


        if _dtype == 'num':
            _X = _X_num
        elif _dtype == 'str':
            _X = _X_str
        else:
            raise Exception


        if X_format == 'pl_series':
            _X = pl.Series(_X[:, 0])
        elif X_format == 'pl_df':
            _X = pl.DataFrame(_X)
        else:
            raise Exception

        # pl wont cast non-str to str, and vice versa.
        # this is a pl casting issue, not pybear.
        if _fill is None or _fill == 'NaN':
            out = _transform(_X, _fill)
        elif (isinstance(_fill , str) and _dtype != 'str') \
            or (not isinstance(_fill , str) and _dtype == 'str'):
            # there is some wack behavior in polars. sometimes it will
            # cast a value of different type to a df/series and other
            # times it wont.
            try:
                out = _transform(_X, _fill)
            except Exception as e:
                # this is handled by polars, let it raise whatever
                pytest.skip(reason=f'cant do more tests without transform')

        else:
            out = _transform(_X, _fill)
        assert isinstance(out, type(_X))
        assert out.shape == _X.shape

        ref = _X.to_numpy().copy()
        ref[nan_mask(ref)] = _fill

        # assertions -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        if _dtype == 'str':
            if len(out.shape) == 1:
                assert np.array_equal(
                    list(map(str, out.to_numpy())),
                    list(map(str, ref))
                )
            else:
                for r_idx in range(out.shape[0]):
                    assert np.array_equal(
                        list(map(str, out.to_numpy()[r_idx])),
                        list(map(str, ref[r_idx]))
                    )
        else:
            assert all(map(
                np.array_equal,
                out.to_numpy(),
                ref,
                (True for _ in out)
            ))




