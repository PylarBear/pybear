# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import warnings
from typing import Literal
from typing_extensions import Union
from uuid import uuid4

import numpy as np
import pandas as pd
import scipy.sparse as ss

from sklearn.feature_selection import VarianceThreshold as VT

from pybear.utilities import nan_mask


# these tests baseline VT handling of various NaN values.
# appears to always return np
# cases where VT wont take nan-like
# 1) np, dtype=str, nan is None
# 2) np & pd, dtype=str or obj, nan is pd.NA


# this is to help in deciding the marginal value of building a
# InterceptManager module to deal with various nans and floating point
# error





bypass = False



@pytest.fixture(scope='module')
def _shape():
    return (10, 3)


@pytest.fixture(scope='module')
def _constant_col(_shape):
    return [1, _shape[1] - 1]


@pytest.fixture(scope='module')
def _base_X(_shape, _constant_col):

    _ = np.random.randint(0,10,_shape)
    _[:, _constant_col] = 1

    return _


@pytest.fixture(scope='module')
def _nan_factory(_base_X, _shape):

    def foo(
        _format: Literal['np', 'pd', 'coo', 'csc', 'csr'],
        _dtype: Literal['flt', 'int', 'str', 'obj'],
        _nan_type: Union[Literal['np', 'pd', 'str', 'None'], None]
    ):

        if _format != 'pd' and _dtype in ['flt', 'int'] and _nan_type == 'pd':
            raise ValueError('numpy numeric array cannot take pd.NA')


        X = _base_X.copy()
        columns = [str(uuid4())[:4] for _ in range(_shape[1])]

        if _dtype == 'flt':
            if _format == 'pd':
                X = pd.DataFrame(data=X, columns=columns).astype(np.float64)
            else:
                X = X.astype(np.float64)
        elif _dtype == 'int':
            if _format == 'pd':
                X = pd.DataFrame(data=X, columns=columns).astype(np.int32)
            else:
                X = X.astype(np.int32)
        elif _dtype == 'str':
            if _format == 'pd':
                X = pd.DataFrame(data=X, columns=columns).astype(str)
            else:
                X = X.astype(str)
        elif _dtype == 'obj':
            if _format == 'pd':
                X = pd.DataFrame(data=X, columns=columns).astype(object)
            else:
                X = X.astype(object)
        else:
            raise Exception

        # _base_choices = [np.nan, None, 'nan', 'NaN', 'NAN']
        # pd df cant take any nan
        # np flt/int can only take the base choices
        # np str/obj can take the base choices and pd.NA

        # _nan_type:Union[Literal['np','pd','str','None'], None]
        if _nan_type == 'np':
            _nan_value = np.nan
        elif _nan_type == 'pd':
            _nan_value = pd.NA
        elif _nan_type == 'str':
            _nan_value = 'nan'
        elif _nan_type == 'None':
            _nan_value = None
        elif _nan_type == None:
            pass  # shouldnt need to declare _nan_value, is bypassed
        else:
            raise Exception


        if _nan_type is not None:

            if _dtype == 'int' and isinstance(X, np.ndarray):
                warnings.warn(
                    f"attempting to put nans into an integer dtype, "
                    f"converted to float"
                )
                X = X.astype(np.float64)

            # if _format == 'pd':
            #     _choices = _base_choices + [pd.NA, '<NA>']
            # else:
            #     if _dtype == 'flt':
            #         _choices = _base_choices
            #     el
            #         X = X.astype(np.float64)
            #         _choices = _base_choices
            #     else:
            #         _choices = _base_choices + [pd.NA]

            # determine how many nans to sprinkle based on _shape and _has_nan
            _sprinkles = max(3, _shape[0] // 10)

            for _c_idx in range(_shape[1]):
                _r_idxs = np.random.choice(
                    range(_shape[0]), _sprinkles, replace=False
                )
                for _r_idx in _r_idxs:
                    if isinstance(X, np.ndarray):
                        X[_r_idx, _c_idx] = _nan_value
                    elif isinstance(X, pd.DataFrame):
                        X.iloc[_r_idx, _c_idx] = _nan_value
                    else:
                        raise Exception

            del _sprinkles

        # do this after sprinkling the nans
        if _format == 'np':
            pass
        elif _format == 'pd':
            pass
        elif _format == 'csc':
            X = ss.csc_array(X)
        elif _format == 'csr':
            X = ss.csr_array(X)
        elif _format == 'coo':
            X = ss.coo_array(X)
        else:
            raise Exception

        return X

    return foo


@pytest.mark.skipif(bypass is True, reason='bypass')
class TestNanFactory:

    @pytest.mark.parametrize('_format', ('np', 'pd', 'csc', 'csr', 'coo'))
    @pytest.mark.parametrize('_dtype', ('flt', 'int', 'str', 'obj'))
    @pytest.mark.parametrize('_nan_type', ('np', 'pd', 'str', 'None', None))
    def test_accuracy(
        self, _nan_factory, _format, _dtype, _nan_type, _constant_col, _shape
    ):

        if _dtype in ['str', 'obj'] and _format in ['csr', 'csc', 'coo']:
            pytest.skip(reason=f"cant put non-num dtype in sparse array")

        if _format != 'pd' and _dtype in ['flt', 'int'] and _nan_type == 'pd':
            # numpy numeric cannot take pd.NA
            with pytest.raises(ValueError):
                out = _nan_factory(
                    _format=_format,
                    _dtype=_dtype,
                    _nan_type=_nan_type
                )
            pytest.skip(reason=f'cant do the tests')
        else:
            out = _nan_factory(
                _format=_format,
                _dtype=_dtype,
                _nan_type=_nan_type
            )

        if _format == 'np':
            assert isinstance(out, np.ndarray)
        elif _format == 'pd':
            assert isinstance(out, pd.DataFrame)
        elif _format == 'csc':
            assert isinstance(out, ss.csc_array)
        elif _format == 'csr':
            assert isinstance(out, ss.csr_array)
        elif _format == 'coo':
            assert isinstance(out, ss.coo_array)
        else:
            raise Exception

        # v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
        # these arent set here, this is just to referee output
        if _nan_type == 'np':
            _nan_value = np.nan
        elif _nan_type == 'pd':
            _nan_value = pd.NA
        elif _nan_type == 'str':
            _nan_value = np.str_('nan')
        elif _nan_type == 'None':
            _nan_value = None
        elif _nan_type == None:
            pass  # shouldnt need to declare _nan_value, is bypassed
        else:
            raise Exception
        # v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^

        if _format in ['csc', 'csr', 'coo']:
            out = out.toarray()

        # everything should be np or pd

        if isinstance(out, np.ndarray):
            if _dtype == 'flt':
                assert out.dtype == np.float64
            elif _dtype == 'int':
                if _format == 'pd' or _nan_type is None:
                    assert out.dtype == np.int32
                else:
                    assert out.dtype == np.float64  # because converted to take nan
            elif _dtype == 'str':
                assert 'U' in str(out.dtype).upper() # str dtype
            elif _dtype == 'obj':
                assert out.dtype == object
            else:
                raise Exception

            # check for nan type
            if _nan_type is not None:
                nans = out[nan_mask(out)].ravel()
                _one_of_the_nans = nans[0]

                if _dtype in ['flt', 'int']:
                    assert type(_one_of_the_nans) == np.float64
                    assert str(_one_of_the_nans) == 'nan'
                elif _dtype == 'str':
                    if _nan_type == 'pd':
                        assert _one_of_the_nans == '<NA>'
                    elif _nan_type == 'None':
                        assert _one_of_the_nans == 'None'
                    else:
                        assert _one_of_the_nans == 'nan'
                elif _dtype == 'obj':
                    if _nan_type == 'None':
                        assert _one_of_the_nans is None
                    elif _nan_type == 'str':
                        assert str(_one_of_the_nans) == 'nan'
                    elif _nan_type == 'pd':
                        assert _one_of_the_nans is pd.NA
                    else:
                        assert type(_one_of_the_nans) == float
                        assert str(_one_of_the_nans) == 'nan'
                else:
                    raise Exception

            # check constant columns
            for idx in range(_shape[1]):
                mask = np.logical_not(nan_mask(out[:, idx]))
                non_nan = out[mask, idx]
                _exp = 1 if _dtype in ['flt', 'int', 'obj'] else '1'
                if idx in _constant_col:
                    assert np.all(non_nan == _exp)
                else:
                    assert not np.all(non_nan == _exp)


        elif isinstance(out, pd.DataFrame):
            for idx, _col in enumerate(out):
                if _dtype == 'flt':
                    if _nan_type == 'str':
                        assert out[_col].dtype == 'O'
                    else:
                        assert out[_col].dtype == np.float64
                elif _dtype == 'int':
                    if _nan_type is None:
                        assert out[_col].dtype == np.int32
                    elif _nan_type == 'str':
                        assert out[_col].dtype == 'O'
                    else:
                        assert out[_col].dtype == np.float64
                elif _dtype == 'str':
                    assert out[_col].dtype == 'O'
                elif _dtype == 'obj':
                    assert out[_col].dtype == object
                else:
                    raise Exception

                # check for nan type
                if _nan_type is not None:
                    nans = out.loc[nan_mask(out[_col]), _col]
                    _one_of_the_nans = nans.iloc[0]

                    if _dtype in ['flt', 'int']:
                        if _nan_type == 'str':
                            assert _one_of_the_nans == 'nan'
                        else:
                            assert type(_one_of_the_nans) == np.float64
                            assert str(_one_of_the_nans) == 'nan'
                    elif _dtype == 'str':
                        if _nan_type == 'np':
                            assert _one_of_the_nans is np.nan
                        elif _nan_type == 'pd':
                            assert _one_of_the_nans is pd.NA
                        elif _nan_type == 'None':
                            assert _one_of_the_nans is None
                        else:
                            assert _one_of_the_nans == 'nan'
                    elif _dtype == 'obj':
                        if _nan_type == 'None':
                            assert _one_of_the_nans is None
                        elif _nan_type == 'str':
                            assert str(_one_of_the_nans) == 'nan'
                        elif _nan_type == 'pd':
                            assert _one_of_the_nans is pd.NA
                        else:
                            assert type(_one_of_the_nans) == float
                            assert str(_one_of_the_nans) == 'nan'
                    else:
                        raise Exception


                # check constant columns
                mask = np.logical_not(nan_mask(out[_col]))
                non_nan = out.loc[mask, _col]
                _exp = 1 if _dtype in ['flt', 'int', 'obj'] else '1'
                if idx in _constant_col:
                    assert np.all(non_nan == _exp)
                else:
                    assert not np.all(non_nan == _exp)

        else:
            raise Exception




class TestVarianceThreshold:

    @pytest.mark.skipif(bypass is True, reason='bypass')
    @pytest.mark.parametrize('_format', ('np', 'pd', 'csc', 'csr', 'coo'))
    @pytest.mark.parametrize('_dtype', ('flt', 'int', 'str', 'obj'))
    @pytest.mark.parametrize('_nan_type', ('np', 'pd', 'str', 'None', None))
    def test_nan_handling(
        self, _nan_factory, _format, _dtype, _nan_type, _constant_col, _shape
    ):

        if _dtype in ['str', 'obj'] and _format in ['csr', 'csc', 'coo']:
            pytest.skip(reason=f"cant put non-num dtype in sparse array")

        if _format != 'pd' and _dtype in ['flt', 'int'] and _nan_type == 'pd':
            pytest.skip(reason=f"numpy numeric cannot take pd.NA")


        X = _nan_factory(
            _format=_format,
            _dtype=_dtype,
            _nan_type=_nan_type
        )

        # cases where VT wont take nan-like
        if (_nan_type == 'pd' and _format in ['np', 'pd'] and
            _dtype in ['str', 'obj']) or \
            (_nan_type == 'None' and _dtype == 'str' and _format == 'np'):

            with pytest.raises(Exception):
                VT(threshold=0).fit(X)
            pytest.skip(reason=f"cant do any tests if wont fit!")
        else:
            trfm = VT(threshold=0).fit(X)


        TRFM_X = trfm.transform(X)
        # appears to always return np

        _X_shape = TRFM_X.shape
        assert _X_shape[0] == _shape[0]
        assert _X_shape[1] == (_shape[1] - len(_constant_col))

        new_c_idx = 0
        for c_idx in range(_shape[1]):

            if c_idx in _constant_col:
                assert trfm.variances_[c_idx] == 0
            else:
                if isinstance(X, np.ndarray):
                    not_nan_mask = np.logical_not(nan_mask(TRFM_X[:, new_c_idx]))
                    assert np.array_equal(
                        X[not_nan_mask, c_idx],
                        TRFM_X[not_nan_mask, new_c_idx]
                    )
                elif isinstance(X, pd.DataFrame):
                    not_nan_mask = np.logical_not(nan_mask(TRFM_X[:, new_c_idx]))
                    assert np.array_equal(
                        X.iloc[not_nan_mask, c_idx],
                        TRFM_X[not_nan_mask, new_c_idx]
                    )
                else:
                    # sparse array
                    not_nan_mask = np.logical_not(
                        nan_mask(TRFM_X.tocsc()[:, [new_c_idx]].toarray().ravel())
                    )
                    assert np.array_equal(
                        X.toarray()[not_nan_mask, c_idx],
                        TRFM_X.toarray()[not_nan_mask, new_c_idx]
                    )

                new_c_idx += 1














