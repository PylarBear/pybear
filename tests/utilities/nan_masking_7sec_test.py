# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.utilities._nan_masking import (
    nan_mask_numerical,
    nan_mask_string,
    nan_mask
)


from uuid import uuid4
import numpy as np
import pandas as pd
import scipy.sparse as ss


import pytest




# 'nan_mask' is tested alongside 'nan_mask_numerical' and
# 'nan_mask_string', it should replicate the behavior of both

# tests numpy arrays, pandas dataframes, and scipy.sparse with various
# nan-like representations (np.nan, 'nan', 'pd.NA', None),
# for float, int, str, and object dtypes.


class Fixtures:


    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (5, 3)


    @staticmethod
    @pytest.fixture(scope='module')
    def _columns(_shape):
        while True:
            _ = np.array([str(uuid4())[:4] for _ in range(_shape[1])])
            if len(_) == len(np.unique(_)):
                return _


    @staticmethod
    @pytest.fixture(scope='module')
    def truth_mask_1(_shape):
        while True:
            __ = np.random.randint(0,2, _shape).astype(bool)
            # make sure that there are both trues and falses in all columns
            for c_idx in range(_shape[1]):
                if 0 < np.sum(__[:, c_idx]) / _shape[0] < 1:
                    pass
                else:
                    break
            else:
                return __


    @staticmethod
    @pytest.fixture(scope='module')
    def truth_mask_2(_shape):
        while True:
            __ = np.random.randint(0,2, _shape).astype(bool)
            # make sure that there are both trues and falses
            for c_idx in range(_shape[1]):
                if 0 < np.sum(__[:, c_idx]) / _shape[0] < 1:
                    pass
                else:
                    break
            else:
                return __


    @staticmethod
    @pytest.fixture(scope='module')
    def pd_assnmt_handle():

        # 24_10_28 pandas future warnings about casting incompatible
        # dtypes. put assignments under a try/except, if OK, return new
        # X, if bad return None. in the test functions, if X comes back
        # as None, skip the test.

        def foo(X: pd.DataFrame, MASK: np.ndarray, value: any):
            try:
                X[MASK] = value
                return X
            except:
                return None

        return foo



class TestNanMaskNumeric(Fixtures):

    # the only np numerical that can take nan is np ndarray.astype(float64)
    # and it must be np.nan, 'nan' (not case sensitive), or None (not pd.NA!)
    @pytest.mark.parametrize('_trial', (1, 2))
    @pytest.mark.parametrize('_nan_type',
        ('npnan', 'strnan', 'any string', 'pdNA', 'none')
    )
    def test_np_float_array(
        self, _shape, truth_mask_1, truth_mask_2, _trial, _nan_type
    ):

        X = np.random.uniform(0,1,_shape)

        if _trial == 1:
            MASK = truth_mask_1
        elif _trial == 2:
            MASK = truth_mask_2
        else:
            raise Exception

        if _nan_type == 'npnan':
            X[MASK] = np.nan
        elif _nan_type == 'strnan':
            X[MASK] = 'nan'
        elif _nan_type == 'any string':
            with pytest.raises(ValueError):
                X[MASK] = 'any string'
            pytest.skip()
        elif _nan_type == 'pdNA':
            with pytest.raises(TypeError):
                X[MASK] = pd.NA
            pytest.skip()
        elif _nan_type == 'none':
            X[MASK] = None
        else:
            raise Exception

        out = nan_mask_numerical(X)
        out_2 = nan_mask(X)

        assert np.array_equal(out, MASK)
        assert np.array_equal(out_2, MASK)



    # numpy integer array cannot take any representation of nan.
    # would need to convert this to float64
    @pytest.mark.parametrize('_trial', (1, 2))
    @pytest.mark.parametrize('_nan_type',
        ('npnan', 'strnan', 'any string', 'pdNA', 'none')
    )
    def test_np_int_array(
        self, _shape, truth_mask_1, truth_mask_2, _trial, _nan_type
    ):

        X = np.random.randint(0,10, _shape)

        if _trial == 1:
            MASK = truth_mask_1
        elif _trial == 2:
            MASK = truth_mask_2
        else:
            raise Exception

        # everything here raises an except
        if _nan_type == 'npnan':
            with pytest.raises(ValueError):
                X[MASK] = np.nan
            pytest.skip()
        elif _nan_type == 'strnan':
            with pytest.raises(ValueError):
                X[MASK] = 'nan'
            pytest.skip()
        elif _nan_type == 'any string':
            with pytest.raises(ValueError):
                X[MASK] = 'any string'
            pytest.skip()
        elif _nan_type == 'pdNA':
            with pytest.raises(TypeError):
                X[MASK] = pd.NA
            pytest.skip()
        elif _nan_type == 'none':
            with pytest.raises(TypeError):
                X[MASK] = None
            pytest.skip()
        else:
            raise Exception

        out = nan_mask_numerical(X)
        out_2 = nan_mask(X)

        assert np.array_equal(out, MASK)
        assert np.array_equal(out_2, MASK)



    # all scipy sparses return np float array np.nans correctly
    @pytest.mark.parametrize('_trial', (1, 2))
    @pytest.mark.parametrize('_format',
        (
            'csr_matrix', 'csc_matrix', 'coo_matrix', 'dia_matrix',
            'lil_matrix', 'dok_matrix', 'bsr_matrix', 'csr_array', 'csc_array',
            'coo_array', 'dia_array', 'lil_array', 'dok_array', 'bsr_array',
        )
    )
    @pytest.mark.parametrize('_nan_type',
        ('npnan', 'strnan', 'any string', 'pdNA', 'none')
    )
    def test_scipy_sparse_via_np_float_array(
        self, _shape, truth_mask_1, truth_mask_2, _trial, _format, _nan_type
    ):

        X = np.random.uniform(0,1,_shape)

        # make a lot of sparsity so that converting to sparse reduces
        for _col_idx in range(_shape[1]):
            _row_idxs = np.random.choice(
                range(_shape[0]),
                _shape[0]//4,
                replace=False
            )
            X[_row_idxs, _col_idx] = 0


        if _trial == 1:
            MASK = truth_mask_1
        elif _trial == 2:
            MASK = truth_mask_2
        else:
            raise Exception

        if _nan_type == 'npnan':
            X[MASK] = np.nan
        elif _nan_type == 'strnan':
            X[MASK] = 'nan'
        elif _nan_type == 'any string':
            with pytest.raises(ValueError):
                X[MASK] = 'any string'
            pytest.skip()
        elif _nan_type == 'pdNA':
            with pytest.raises(TypeError):
                X[MASK] = pd.NA
            pytest.skip()
        elif _nan_type == 'none':
            X[MASK] = None
        else:
            raise Exception


        if _format == 'csr_matrix':
            X_wip = ss._csr.csr_matrix(X)
        elif _format == 'csc_matrix':
            X_wip = ss._csc.csc_matrix(X)
        elif _format == 'coo_matrix':
            X_wip = ss._coo.coo_matrix(X)
        elif _format == 'dia_matrix':
            X_wip = ss._dia.dia_matrix(X)
        elif _format == 'lil_matrix':
            X_wip = ss._lil.lil_matrix(X)
        elif _format == 'dok_matrix':
            X_wip = ss._dok.dok_matrix(X)
        elif _format == 'bsr_matrix':
            X_wip = ss._bsr.bsr_matrix(X)
        elif _format == 'csr_array':
            X_wip = ss._csr.csr_array(X)
        elif _format == 'csc_array':
            X_wip = ss._csc.csc_array(X)
        elif _format == 'coo_array':
            X_wip = ss._coo.coo_array(X)
        elif _format == 'dia_array':
            X_wip = ss._dia.dia_array(X)
        elif _format == 'lil_array':
            X_wip = ss._lil.lil_array(X)
        elif _format == 'dok_array':
            X_wip = ss._dok.dok_array(X)
        elif _format == 'bsr_array':
            X_wip = ss._bsr.bsr_array(X)
        else:
            raise Exception


        # covert back to np to see if nan mask was affected
        X = X_wip.toarray()

        out = nan_mask_numerical(X)
        out_2 = nan_mask(X)

        assert np.array_equal(out, MASK)
        assert np.array_equal(out_2, MASK)



    # pd float dfs can take any of the following representations of nan
    # and convert them to either np.nan or pd.NA:
    # np.nan, any string, pd.NA, or None
    @pytest.mark.parametrize('_trial', (1, 2))
    @pytest.mark.parametrize('_nan_type',
        ('npnan', 'strnan', 'any string', 'pdNA', 'none')
    )
    def test_pd_float(
        self, _shape, truth_mask_1, truth_mask_2, _trial, _nan_type, _columns,
        pd_assnmt_handle
    ):

        X = pd.DataFrame(
            data = np.random.uniform(0, 1, _shape),
            columns = _columns,
            dtype=np.float64
        )


        if _trial == 1:
            MASK = truth_mask_1
        elif _trial == 2:
            MASK = truth_mask_2
        else:
            raise Exception

        if _nan_type == 'npnan':
            X = pd_assnmt_handle(X, MASK, np.nan)
        elif _nan_type == 'strnan':
            X = pd_assnmt_handle(X, MASK, 'nan')
        elif _nan_type == 'any string':
            X = pd_assnmt_handle(X, MASK, 'any string')
        elif _nan_type == 'pdNA':
            X = pd_assnmt_handle(X, MASK, pd.NA)
        elif _nan_type == 'none':
            X = pd_assnmt_handle(X, MASK, None)
        else:
            raise Exception

        # if exception is raised by pd_assnmnt_handle because of casting
        # nan to disallowed dtype, then X is None and skip test
        if X is None:
            pytest.skip(reason=f"invalid value cast to dataframe dtype, skip test")

        if _nan_type == 'any string':
            with pytest.raises(ValueError):
                nan_mask_numerical(X)

            assert np.sum(nan_mask(X)) == 0
        else:
            assert np.array_equal(nan_mask_numerical(X), MASK)
            assert np.array_equal(nan_mask(X), MASK)


    # pd int dfs can take any of the following representations of nan
    # and convert them to either np.nan or pd.NA:
    # np.nan, any string, pd.NA, or None
    @pytest.mark.parametrize('_trial', (1, 2))
    @pytest.mark.parametrize('_nan_type',
        ('npnan', 'strnan', 'any string', 'pdNA', 'none')
    )
    def test_pd_int(
        self, _shape, truth_mask_1, truth_mask_2, _trial, _nan_type, _columns,
        pd_assnmt_handle
    ):

        X = pd.DataFrame(
            data = np.random.randint(0, 10, _shape),
            columns = _columns,
            dtype=np.uint32
        )


        if _trial == 1:
            MASK = truth_mask_1
        elif _trial == 2:
            MASK = truth_mask_2
        else:
            raise Exception

        if _nan_type == 'npnan':
            X = pd_assnmt_handle(X, MASK, np.nan)
        elif _nan_type == 'strnan':
            X = pd_assnmt_handle(X, MASK, 'nan')
        elif _nan_type == 'any string':
            X = pd_assnmt_handle(X, MASK, 'any string')
        elif _nan_type == 'pdNA':
            X = pd_assnmt_handle(X, MASK, pd.NA)
        elif _nan_type == 'none':
            X = pd_assnmt_handle(X, MASK, None)
        else:
            raise Exception

        # if exception is raised by pd_assnmnt_handle because of casting
        # nan to disallowed dtype, then X is None and skip test
        if X is None:
            pytest.skip(reason=f"invalid value cast to dataframe dtype, skip test")

        # putting float nans into a pd int df
        # coerces the df over to float64
        if _nan_type in ['strnan', 'any string']:
            assert list(set(X.dtypes))[0] == 'O'
        else:
            assert list(set(X.dtypes))[0] == np.float64

        if _nan_type == 'any string':
            with pytest.raises(ValueError):
                nan_mask_numerical(X)

            assert np.sum(nan_mask(X)) == 0
        else:
            assert np.array_equal(nan_mask_numerical(X), MASK)
            assert np.array_equal(nan_mask(X), MASK)


    # make a pandas dataframe with all the possible things that make an
    # nan-like in a dataframe. see if converting to np array converts
    # them all to np.nan, and is recognized.
    # takeaway:
    # to_numpy() converts all of these different nans to np.nan correctly
    def numpy_float_via_pd_made_with_various_nan_types(
        self, _shape, truth_mask_1, truth_mask_2, _trial, _nan_type, _columns
    ):

        X = pd.DataFrame(
            data = np.random.uniform(0, 1, _shape),
            columns = _columns,
            dtype=np.float64
        )

        _pool = (np.nan, 'nan', 'any string', pd.NA, None, '<NA>')

        # sprinkle the various nan-types into the float DF
        # make a mask DF to mark the places of the nan-likes
        MASK = np.zeros(_shape).astype(bool)
        for _sprinkling_itr in range(X.size//10):
            _row_idx = np.random.randint(_shape[0])
            _col_idx = np.random.randint(_shape[1])

            X.iloc[_row_idx, _col_idx] = np.random.choice(_pool)
            MASK[_row_idx, _col_idx] = True

        X = X.to_numpy()

        out = nan_mask_numerical(X)
        out_2 = nan_mask(X)

        assert np.array_equal(out, MASK)
        assert np.array_equal(out_2, MASK)


    def test_nan_mask_numerical_takes_str_numbers(self):

        # 1D -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        _X = list(map(str, list(range(10))))

        assert all(map(isinstance, _X, (str for _ in _X)))
        np.array(_X, dtype=np.float64)

        _X[1] = 'nan'
        _X[2] = np.nan

        ref = np.zeros((10,)).astype(bool)
        ref[1] = True
        ref[2] = True
        ref = ref.tolist()

        out = nan_mask_numerical(_X)

        assert np.array_equal(out, ref)
        # END 1D -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # 2D -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        _X = np.random.randint(0, 10, (5, 3)).astype(str)

        assert all(map(isinstance, _X[0], (str for _ in _X[0])))
        np.array(_X, dtype=np.float64)

        _X[1][0] = 'nan'
        _X[2][0] = np.nan

        ref = np.zeros((5, 3)).astype(bool)
        ref[1][0] = True
        ref[2][0] = True
        ref = ref.tolist()

        out = nan_mask_numerical(_X)

        assert np.array_equal(out, ref)

        # END 2D -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    def test_nan_mask_numerical_takes_weird_str_numbers(self):

        # 25_04_12 there was a problem with all of the Text modules failing
        # on string data that looks like ['8\n', '9\n']. Only when there are
        # no other legit strings. the problem appears to be in
        # nan_mask_numerical.

        _nums = ['1\n', '2\n', '3\n', '4\n','5\n', '6\n', '7\n']

        # use this as a control to show this works on string
        assert not any(nan_mask_string(_nums))

        assert not any(nan_mask_numerical(_nums))



class TestNanMaskString(Fixtures):

    # scipy sparse cannot take non-numeric datatypes

    @staticmethod
    @pytest.fixture()
    def _X(_shape):
        return np.random.choice(list('abcdefghij'), _shape, replace=True)


    # np str arrays can take any of the following representations of nan:
    # np.nan, 'nan', pd.NA, None
    @pytest.mark.parametrize('_trial', (1, 2))
    @pytest.mark.parametrize('_nan_type',
        ('npnan', 'strnan', 'any string', 'pdNA', 'none')
    )
    def test_np_array_str(
        self, _X, truth_mask_1, truth_mask_2, _trial, _nan_type, _shape
    ):
        # remember to set str dtype like '<U10' to _X

        X = _X.astype('<U10')

        if _trial == 1:
            MASK = truth_mask_1
        elif _trial == 2:
            MASK = truth_mask_2
        else:
            raise Exception

        if _nan_type == 'npnan':
            X[MASK] = np.nan
        elif _nan_type == 'strnan':
            X[MASK] = 'nan'
        elif _nan_type == 'any string':
            X[MASK] = 'any string'
            # this is a valid assignment into a str array
        elif _nan_type == 'pdNA':
            X[MASK] = pd.NA
        elif _nan_type == 'none':
            X[MASK] = None
        else:
            raise Exception

        out = nan_mask_string(X)
        out_2 = nan_mask(X)

        if _nan_type == 'any string':
            assert np.sum(out) == 0
            assert np.sum(out_2) == 0
        else:
            assert np.array_equal(out, MASK)
            assert np.array_equal(out_2, MASK)


    # np object arrays can take any of the following representations of nan:
    # np.nan, 'nan', pd.NA, None
    @pytest.mark.parametrize('_trial', (1, 2))
    @pytest.mark.parametrize('_nan_type',
        ('npnan', 'strnan', 'any string', 'pdNA', 'none')
    )
    def test_np_array_object(
        self, _X, truth_mask_1, truth_mask_2, _trial, _nan_type, _shape
    ):
        # remember to set object dtype to _X
        X = _X.astype(object)

        if _trial == 1:
            MASK = truth_mask_1
        elif _trial == 2:
            MASK = truth_mask_2
        else:
            raise Exception

        if _nan_type == 'npnan':
            X[MASK] = np.nan
        elif _nan_type == 'strnan':
            X[MASK] = 'nan'
        elif _nan_type == 'any string':
            X[MASK] = 'any string'
            # this is a valid assignment into a obj array
        elif _nan_type == 'pdNA':
            X[MASK] = pd.NA
        elif _nan_type == 'none':
            X[MASK] = None
        else:
            raise Exception

        out = nan_mask_string(X)
        out_2 = nan_mask(X)

        if _nan_type == 'any string':
            assert np.sum(out) == 0
            assert np.sum(out_2) == 0
        else:
            assert np.array_equal(out, MASK)
            assert np.array_equal(out_2, MASK)


    # pd str dfs can take any of the following representations of nan:
    # np.nan, 'nan', pd.NA, None
    @pytest.mark.parametrize('_trial', (1, 2))
    @pytest.mark.parametrize('_nan_type',
        ('npnan', 'strnan', 'any string', 'pdNA', 'none')
    )
    def test_pd_str(
        self, _X, truth_mask_1, truth_mask_2, _trial, _nan_type, _shape,
        _columns, pd_assnmt_handle
    ):
        # remember to set str dtype like '<U10' to _X

        X = pd.DataFrame(
            data = _X,
            columns = _columns,
            dtype='<U10'
        )

        if _trial == 1:
            MASK = truth_mask_1
        elif _trial == 2:
            MASK = truth_mask_2
        else:
            raise Exception

        if _nan_type == 'npnan':
            X = pd_assnmt_handle(X, MASK, np.nan)
        elif _nan_type == 'strnan':
            X = pd_assnmt_handle(X, MASK, 'nan')
        elif _nan_type == 'any string':
            X = pd_assnmt_handle(X, MASK, 'any string')
            # this is a valid assignment into a pd str type
        elif _nan_type == 'pdNA':
            X = pd_assnmt_handle(X, MASK, pd.NA)
        elif _nan_type == 'none':
            X = pd_assnmt_handle(X, MASK, None)
        else:
            raise Exception

        # if exception is raised by pd_assnmnt_handle because of casting
        # nan to disallowed dtype, then X is None and skip test
        if X is None:
            pytest.skip(reason=f"invalid value cast to dataframe dtype, skip test")

        out = nan_mask_string(X)
        out_2 = nan_mask(X)

        if _nan_type == 'any string':
            assert np.sum(out) == 0
            assert np.sum(out_2) == 0
        else:
            assert np.array_equal(out, MASK)
            assert np.array_equal(out_2, MASK)


    # pd obj dfs can take any of the following representations of nan:
    # np.nan, 'nan', pd.NA, None
    @pytest.mark.parametrize('_trial', (1, 2))
    @pytest.mark.parametrize('_nan_type',
        ('npnan', 'strnan', 'any string', 'pdNA', 'none')
    )
    def test_pd_object(
        self, _X, truth_mask_1, truth_mask_2, _trial, _nan_type, _shape,
        _columns, pd_assnmt_handle
    ):
        # remember to set object dtype to _X
        X = pd.DataFrame(
            data = _X,
            columns = _columns,
            dtype=object
        )

        if _trial == 1:
            MASK = truth_mask_1
        elif _trial == 2:
            MASK = truth_mask_2
        else:
            raise Exception

        if _nan_type == 'npnan':
            X = pd_assnmt_handle(X, MASK, np.nan)
        elif _nan_type == 'strnan':
            X = pd_assnmt_handle(X, MASK, 'nan')
        elif _nan_type == 'any string':
            X = pd_assnmt_handle(X, MASK, 'any string')
            # this is a valid assignment into a pd object type
        elif _nan_type == 'pdNA':
            X = pd_assnmt_handle(X, MASK, pd.NA)
        elif _nan_type == 'none':
            X = pd_assnmt_handle(X, MASK, None)
        else:
            raise Exception

        # if exception is raised by pd_assnmnt_handle because of casting
        # nan to disallowed dtype, then X is None and skip test
        if X is None:
            pytest.skip(reason=f"invalid value cast to dataframe dtype, skip test")

        out = nan_mask_string(X)
        out_2 = nan_mask(X)

        if _nan_type == 'any string':
            assert np.sum(out) == 0
            assert np.sum(out_2) == 0
        else:
            assert np.array_equal(out, MASK)
            assert np.array_equal(out_2, MASK)












