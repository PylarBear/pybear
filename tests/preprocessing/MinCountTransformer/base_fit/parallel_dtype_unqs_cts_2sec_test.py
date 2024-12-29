# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np

from pybear.preprocessing.MinCountTransformer._base_fit. \
    _parallel_dtypes_unqs_cts import _dtype_unqs_cts_processing




# def _dtype_unqs_cts_processing(
#         _column_of_X,
#         col_idx: int
#     )



class TestParallelizedRowMasks:

    @staticmethod
    @pytest.fixture
    def _rows():
        return 100


    @staticmethod
    @pytest.fixture
    def _pool_size(_rows):
        return _rows // 20


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



    @staticmethod
    @pytest.fixture
    def good_unq_ct_dict():

        def foo(any_column):
            return dict((zip(*np.unique(any_column, return_counts=True))))

        return foo



    def test_accuracy_dtype_unq_ct__str(self,
            str_column_no_nan,
            str_column_nan,
            good_unq_ct_dict
        ):


        for _dtype, column_type in \
                (('obj',str_column_no_nan), ('obj', str_column_nan)):


            out_dtype, out_unq_ct_dict = _dtype_unqs_cts_processing(
                column_type,
                col_idx=0
            )

            assert out_dtype == 'obj'

            EXP = good_unq_ct_dict(column_type)
            EXP_KEYS = np.fromiter(EXP.keys(), dtype='<U1')
            EXP_VALUES = np.fromiter(EXP.values(), dtype=np.uint16)
            OUT_KEYS = np.fromiter(out_unq_ct_dict.keys(), dtype='<U1')
            OUT_VALUES = np.fromiter(out_unq_ct_dict.values(), dtype=np.uint16)


            assert np.array_equiv(OUT_KEYS, EXP_KEYS)
            assert np.array_equiv(OUT_VALUES, EXP_VALUES)


    def test_accuracy_dtype_unq_ct__float(self,
            float_column_no_nan,
            float_column_nan,
            good_unq_ct_dict
        ):

        for _dtype, column_type in \
                (('float', float_column_no_nan), ('float', float_column_nan)):

            out_dtype, out_unq_ct_dict = _dtype_unqs_cts_processing(
                column_type,
                col_idx=0
            )

            assert out_dtype == 'float'

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



    def test_accuracy_dtype_unq_ct__int(self,
            int_column_no_nan,
            int_column_nan,
            good_unq_ct_dict
        ):

        for _dtype, column_type in \
                (('int', int_column_no_nan), ('int', int_column_nan)):

            out_dtype, out_unq_ct_dict = _dtype_unqs_cts_processing(
                column_type,
                col_idx=0
            )

            assert out_dtype == 'int'

            EXP = good_unq_ct_dict(column_type)
            EXP_KEYS = np.fromiter(EXP.keys(), dtype=np.uint32)
            EXP_VALUES = np.fromiter(EXP.values(), dtype=np.uint16)
            OUT_KEYS = np.fromiter(out_unq_ct_dict.keys(), dtype=np.uint32)
            OUT_VALUES = np.fromiter(out_unq_ct_dict.values(), dtype=np.uint16)

            assert np.allclose(OUT_KEYS, EXP_KEYS, rtol=1e-6)

            assert np.array_equiv(OUT_VALUES, EXP_VALUES)











