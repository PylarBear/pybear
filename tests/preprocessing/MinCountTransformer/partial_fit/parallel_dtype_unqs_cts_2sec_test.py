# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np
import scipy.sparse as ss

from pybear.preprocessing.MinCountTransformer._partial_fit. \
    _parallel_dtypes_unqs_cts import _parallel_dtypes_unqs_cts



class TestParallelizedRowMasks:


    # @joblib.wrap_non_picklable_objects
    # def _parallel_dtypes_unqs_cts(
    #     _column_of_X: npt.NDArray[DataType],
    #     _n_rows: int,
    #     _col_idx: int
    # ) -> tuple[str, dict[DataType, int]]:


    # fixtures ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    @staticmethod
    @pytest.fixture
    def _rows():
        return 100


    @staticmethod
    @pytest.fixture
    def _pool_size(_rows):
        return _rows // 20

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    @staticmethod
    @pytest.fixture
    def float_column_no_nan(_pool_size, _rows):
        return np.random.uniform(0, _pool_size, (_rows, 1)).astype(np.float64)


    @staticmethod
    @pytest.fixture
    def float_column_nan(float_column_no_nan, _rows):

        NAN_MASK = np.random.choice(np.arange(_rows), int(_rows // 10),
                                    replace=False)
        float_column_nan = float_column_no_nan.copy()
        float_column_nan[NAN_MASK, 0] = np.nan

        return float_column_nan
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    @staticmethod
    @pytest.fixture
    def int_column_no_nan(_pool_size, _rows):
        return np.random.randint(0, _pool_size, (_rows, 1))


    @staticmethod
    @pytest.fixture
    def int_column_nan(int_column_no_nan, _rows):

        NAN_MASK = np.random.choice(np.arange(_rows), int(_rows // 10),
                                    replace=False)
        int_column_nan = int_column_no_nan.copy().astype(np.float64)
        int_column_nan[NAN_MASK, 0] = np.nan

        return int_column_nan
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    @staticmethod
    @pytest.fixture
    def str_column_no_nan(_pool_size, _rows):

        pool = list('abcdefghijklmnopqrstuvwxyz')[:_pool_size]
        return np.random.choice(pool, _rows, replace=True).reshape((-1, 1))


    @staticmethod
    @pytest.fixture
    def str_column_nan(str_column_no_nan, _rows):

        NAN_MASK = np.random.choice(np.arange(_rows), int(_rows // 10),
                                    replace=False)
        str_column_nan = str_column_no_nan.copy().astype('<U3')
        str_column_nan[NAN_MASK, 0] = 'nan'
        return str_column_nan
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --



    @staticmethod
    @pytest.fixture
    def good_unq_ct_dict():

        def foo(any_column):
            return dict((zip(*np.unique(any_column, return_counts=True))))

        return foo

    # END fixtures ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **


    # v v v v v v v v v v v v  TESTS  v v v v v v v v v v v v v v v v v v

    def test_accuracy_dtype_unq_ct__str(
        self,
        str_column_no_nan,
        str_column_nan,
        good_unq_ct_dict,
        _rows
    ):

        for _dtype, column_type in \
                (('obj',str_column_no_nan), ('obj', str_column_nan)):

            out_dtype, out_unq_ct_dict = _parallel_dtypes_unqs_cts(
                column_type,
                _rows,
                _col_idx=0
            )

            assert out_dtype == _dtype

            EXP = good_unq_ct_dict(column_type)
            EXP_KEYS = np.fromiter(EXP.keys(), dtype='<U1')
            EXP_VALUES = np.fromiter(EXP.values(), dtype=np.uint16)
            OUT_KEYS = np.fromiter(out_unq_ct_dict.keys(), dtype='<U1')
            OUT_VALUES = np.fromiter(out_unq_ct_dict.values(), dtype=np.uint16)


            assert np.array_equiv(OUT_KEYS, EXP_KEYS)
            assert np.array_equiv(OUT_VALUES, EXP_VALUES)


    def test_accuracy_dtype_unq_ct__float(
        self,
        float_column_no_nan,
        float_column_nan,
        good_unq_ct_dict,
        _rows
    ):

        for _dtype, column_type in \
                (('float', float_column_no_nan), ('float', float_column_nan)):

            out_dtype, out_unq_ct_dict = _parallel_dtypes_unqs_cts(
                column_type,
                _rows,
                _col_idx=0
            )

            assert out_dtype == _dtype

            EXP = good_unq_ct_dict(column_type)
            EXP_KEYS = np.fromiter(EXP.keys(), dtype=np.float64)
            EXP_VALUES = np.fromiter(EXP.values(), dtype=np.uint16)
            OUT_KEYS = np.fromiter(out_unq_ct_dict.keys(), dtype=np.float64)
            OUT_VALUES = np.fromiter(out_unq_ct_dict.values(), dtype=np.uint16)

            if any(np.isnan(EXP_KEYS)):
                MASK = np.logical_not(np.isnan(EXP_KEYS))
                EXP_KEYS = EXP_KEYS[MASK]
                OUT_KEYS = OUT_KEYS[MASK]

            assert np.allclose(OUT_KEYS, EXP_KEYS, rtol=1e-6)

            assert np.array_equiv(OUT_VALUES, EXP_VALUES)


    def test_accuracy_dtype_unq_ct__int(
        self,
        int_column_no_nan,
        int_column_nan,
        good_unq_ct_dict,
        _rows
    ):

        for _dtype, column_type in \
                (('int', int_column_no_nan), ('int', int_column_nan)):

            out_dtype, out_unq_ct_dict = _parallel_dtypes_unqs_cts(
                column_type,
                _rows,
                _col_idx=0
            )

            assert out_dtype == 'int'

            EXP = good_unq_ct_dict(column_type)
            EXP_KEYS = np.fromiter(EXP.keys(), dtype=np.uint32)
            EXP_VALUES = np.fromiter(EXP.values(), dtype=np.uint16)
            OUT_KEYS = np.fromiter(out_unq_ct_dict.keys(), dtype=np.uint32)
            OUT_VALUES = np.fromiter(out_unq_ct_dict.values(), dtype=np.uint16)

            assert np.allclose(OUT_KEYS, EXP_KEYS, rtol=1e-6)

            assert np.array_equiv(OUT_VALUES, EXP_VALUES)


    # coo, dia, bsr cannot be sliced
    @pytest.mark.parametrize('_format', ('csr', 'csc', 'lil', 'dok'))
    @pytest.mark.parametrize('_dtype', ('float', 'int'))
    @pytest.mark.parametrize('_has_nan', (True, False))
    def test_accuracy_scipy_sparse(
        self,
        good_unq_ct_dict,
        _rows,
        _format,
        _dtype,
        _has_nan
    ):

        # _column_getter() sends the .data attribute of the column only.
        # 1s are missing and need to be inferred by _p_d_u_c().

        # doctor a 1D column with some zeros so scipy takes them out.
        if _dtype == 'int':
            while True:
                _X_np = np.random.randint(0, 3, (_rows,)).astype(np.float64)
                if np.any((_X_np == 0)) and len(np.unique(_X_np)) >= 3:
                    break
        elif _dtype == 'float':
            _X_np = np.random.uniform(0, 1, (_rows,))
            _rand_idxs = \
                np.random.choice(range(_rows), _rows // 10, replace=False)
            _X_np[_rand_idxs] = 0
        else:
            raise Exception

        # END doctor a 1D column -- -- -- -- -- -- -- -- -- -- -- -- --

        if _has_nan:
            _rand_idxs = np.random.choice(range(_rows), _rows//10, replace=False)
            _X_np[_rand_idxs] = np.nan

        _X_np = _X_np.reshape((-1, 1))

        # convert the np array to a scipy sparse -- -- -- -- -- -- -- -- --
        # _column_getter() (what is extracting the column that is fed into here)
        # does it like: column = _X[:, [_col_idx]].tocsc().data

        if _format == 'csr':
            X = ss.csr_array(_X_np)[:, [0]].tocsc().data
        elif _format == 'csc':
            X = ss.csc_array(_X_np)[:, [0]].tocsc().data
        elif _format == 'lil':
            X = ss.lil_array(_X_np)[:, [0]].tocsc().data
        elif _format == 'dok':
            X = ss.dok_array(_X_np)[:, [0]].tocsc().data
        else:
            raise Exception
        # END convert the np array to a scipy sparse -- -- -- -- -- -- --

        out_dtype, out_unq_ct_dict = \
            _parallel_dtypes_unqs_cts(
                X,
                _rows,
                _col_idx=0
            )

        if _dtype == 'int':
            assert out_dtype == 'int'
        elif _dtype == 'float':
            assert out_dtype == 'float'

        EXP = good_unq_ct_dict(_X_np)
        EXP_KEYS = np.fromiter(EXP.keys(), dtype=np.uint32)
        EXP_VALUES = np.fromiter(EXP.values(), dtype=np.uint16)
        OUT_KEYS = np.fromiter(out_unq_ct_dict.keys(), dtype=np.uint32)
        OUT_VALUES = np.fromiter(out_unq_ct_dict.values(), dtype=np.uint16)

        assert np.allclose(OUT_KEYS, EXP_KEYS, rtol=1e-6)

        assert np.array_equiv(OUT_VALUES, EXP_VALUES)













