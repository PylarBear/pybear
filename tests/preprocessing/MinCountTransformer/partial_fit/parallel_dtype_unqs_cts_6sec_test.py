# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.preprocessing._MinCountTransformer._partial_fit. \
    _parallel_dtypes_unqs_cts import _parallel_dtypes_unqs_cts



class TestParallelizedDtypeUnqsCts:


    # @joblib.wrap_non_picklable_objects
    # def _parallel_dtypes_unqs_cts(
    #     _chunk_of_X: npt.NDArray,
    # ) -> list[tuple[str, dict[Any, int]]]:


    # fixtures ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    # pizza
    # @staticmethod
    # @pytest.fixture(scope='module')
    # def _shape():
    #     return (100, 10)


    @staticmethod
    @pytest.fixture(scope='module')
    def _pool_size(_shape):
        return _shape[0] // 20


    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    @staticmethod
    @pytest.fixture(scope='module')
    def float_chunk_no_nan(_pool_size, _shape):
        return np.random.uniform(0, _pool_size, _shape).astype(np.float64)


    @staticmethod
    @pytest.fixture(scope='module')
    def float_chunk_nan(float_chunk_no_nan, _shape):

        float_chunk_nan = float_chunk_no_nan.copy()

        _rows = np.arange(_shape[0])
        _num_nans = int(_shape[0] // 10)

        for _c_idx in range(_shape[1]):
            float_chunk_nan[
                np.random.choice(_rows, _num_nans, replace=False), _c_idx
            ] = np.nan

        del _rows, _num_nans

        return float_chunk_nan
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    @staticmethod
    @pytest.fixture(scope='module')
    def int_chunk_no_nan(_pool_size, _shape):

        return np.random.randint(0, _pool_size, _shape)


    @staticmethod
    @pytest.fixture(scope='module')
    def int_chunk_nan(int_chunk_no_nan, _shape):

        int_chunk_nan = int_chunk_no_nan.copy().astype(np.float64)

        _rows = np.arange(_shape[0])
        _num_nans = int(_shape[0] // 10)

        for _c_idx in range(_shape[1]):

            int_chunk_nan[
                np.random.choice(_rows, _num_nans, replace=False), _c_idx
            ] = np.nan

        del _rows, _num_nans

        return int_chunk_nan
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    @staticmethod
    @pytest.fixture(scope='module')
    def str_chunk_no_nan(_pool_size, _shape):

        pool = list('abcdefghijklmnopqrstuvwxyz')[:_pool_size]

        return np.random.choice(pool, _shape, replace=True)


    @staticmethod
    @pytest.fixture(scope='module')
    def str_chunk_nan(str_chunk_no_nan, _shape):

        str_chunk_nan = str_chunk_no_nan.copy().astype('<U3')

        _rows = np.arange(_shape[0])
        _num_nans = int(_shape[0] // 10)

        for _c_idx in range(_shape[1]):
            str_chunk_nan[
                np.random.choice(_rows, _num_nans, replace=False), _c_idx
            ] = 'nan'

        del _rows, _num_nans

        return str_chunk_nan
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


    @staticmethod
    @pytest.fixture(scope='module')
    def good_unq_ct_dicts():

        def foo(any_chunk):

            list_of_dicts = []
            for _c_idx in range(any_chunk.shape[1]):
                list_of_dicts.append(
                    dict((zip(*np.unique(any_chunk[:, _c_idx], return_counts=True))))
                )

            return list_of_dicts

        return foo

    # END fixtures ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **


    def test_accuracy_dtype_unq_ct__str(
        self, str_chunk_no_nan, str_chunk_nan, good_unq_ct_dicts
    ):

        for _exp_dtype, _chunk in \
                (('obj',str_chunk_no_nan), ('obj', str_chunk_nan)):

            out_dtypes_unq_ct_dicts = _parallel_dtypes_unqs_cts(_chunk)
            exp_unq_ct_dicts = good_unq_ct_dicts(_chunk)

            for _c_idx, (_out_dtype, _out_unq_ct_dict) in \
                    enumerate(out_dtypes_unq_ct_dicts):

                assert _out_dtype == _exp_dtype

                EXP = exp_unq_ct_dicts[_c_idx]
                EXP_KEYS = np.fromiter(EXP.keys(), dtype='<U1')
                EXP_VALUES = np.fromiter(EXP.values(), dtype=np.uint16)
                OUT_KEYS = np.fromiter(_out_unq_ct_dict.keys(), dtype='<U1')
                OUT_VALUES = np.fromiter(_out_unq_ct_dict.values(), dtype=np.uint16)

                assert np.array_equiv(OUT_KEYS, EXP_KEYS)
                assert np.array_equiv(OUT_VALUES, EXP_VALUES)


    def test_accuracy_dtype_unq_ct__float(
        self, float_chunk_no_nan, float_chunk_nan, good_unq_ct_dicts
    ):

        for _exp_dtype, _chunk in \
                (('float', float_chunk_no_nan), ('float', float_chunk_nan)):

            out_dtypes_unq_ct_dicts = _parallel_dtypes_unqs_cts(_chunk)
            exp_unq_ct_dicts = good_unq_ct_dicts(_chunk)

            for _c_idx, (_out_dtype, _out_unq_ct_dict) in \
                    enumerate(out_dtypes_unq_ct_dicts):

                assert _out_dtype == _exp_dtype

                EXP = exp_unq_ct_dicts[_c_idx]
                EXP_KEYS = np.fromiter(EXP.keys(), dtype=np.float64)
                EXP_VALUES = np.fromiter(EXP.values(), dtype=np.uint16)
                OUT_KEYS = np.fromiter(_out_unq_ct_dict.keys(), dtype=np.float64)
                OUT_VALUES = np.fromiter(_out_unq_ct_dict.values(), dtype=np.uint16)

                if any(np.isnan(EXP_KEYS)):
                    MASK = np.logical_not(np.isnan(EXP_KEYS))
                    EXP_KEYS = EXP_KEYS[MASK]
                    OUT_KEYS = OUT_KEYS[MASK]

                assert np.allclose(OUT_KEYS, EXP_KEYS, rtol=1e-6)

                assert np.array_equiv(OUT_VALUES, EXP_VALUES)


    def test_accuracy_dtype_unq_ct__int(
        self, int_chunk_no_nan, int_chunk_nan, good_unq_ct_dicts
    ):

        for _exp_dtype, _chunk in \
                (('int', int_chunk_no_nan), ('int', int_chunk_nan)):

            out_dtypes_unq_ct_dicts = _parallel_dtypes_unqs_cts(_chunk)
            exp_unq_ct_dicts = good_unq_ct_dicts(_chunk)

            for _c_idx, (_out_dtype, _out_unq_ct_dict) in \
                    enumerate(out_dtypes_unq_ct_dicts):

                assert _out_dtype == _exp_dtype

                EXP = exp_unq_ct_dicts[_c_idx]
                EXP_KEYS = np.fromiter(EXP.keys(), dtype=np.uint32)
                EXP_VALUES = np.fromiter(EXP.values(), dtype=np.uint16)
                OUT_KEYS = np.fromiter(_out_unq_ct_dict.keys(), dtype=np.uint32)
                OUT_VALUES = np.fromiter(_out_unq_ct_dict.values(), dtype=np.uint16)

                assert np.allclose(OUT_KEYS, EXP_KEYS, rtol=1e-6)

                assert np.array_equiv(OUT_VALUES, EXP_VALUES)




