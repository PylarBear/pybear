# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest
import numpy as np
from pybear.sparse_dict._linalg import (
                                        sparse_identity
)


class TestSparseIdentity:


    # outer_len ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('non_num',
         (True, False, None, [1], (1,), {1}, {'a':1}, min, int, lambda x: x, 'a')
    )
    def test_rejects_non_numeric_outer_len(self, non_num):
        with pytest.raises(TypeError):
            sparse_identity(non_num, 10, dtype=int)


    @pytest.mark.parametrize('non_int', (np.pi, np.nan, 4.234))
    def test_rejects_float_outer_len(self, non_int):
        with pytest.raises(TypeError):
            sparse_identity(non_int, 10, dtype=int)


    @pytest.mark.parametrize('low_outer_len', (0, -1))
    def test_rejects_outer_len_lt_1(self, low_outer_len):
        with pytest.raises(ValueError):
            sparse_identity(low_outer_len, 10, dtype=int)

    # END outer_len ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # inner_len ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('non_num',
         (True, False, None, [1], (1,), {1}, {'a':1}, min, int, lambda x: x, 'a')
    )
    def test_rejects_non_numeric_inner_len(self, non_num):
        with pytest.raises(TypeError):
            sparse_identity(10, non_num, dtype=int)


    @pytest.mark.parametrize('non_int', (np.pi, np.nan, 4.234))
    def test_rejects_float_inner_len(self, non_int):
        with pytest.raises(TypeError):
            sparse_identity(10, non_int, dtype=int)


    @pytest.mark.parametrize('low_inner_len', (0, -1))
    def test_rejects_inner_len_lt_1(self, low_inner_len):
        with pytest.raises(ValueError):
            sparse_identity(10, low_inner_len, dtype=int)

    # END inner_len ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    @pytest.mark.parametrize('_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    def test_accepts_good_dtype(self, _dtype):

        out = sparse_identity(5,5,_dtype)
        assert type(out[0][0]) is _dtype


    @pytest.mark.parametrize('bad_dtype',
         (0, 1, True, False, None, np.pi, float('inf'), [1], (1,), {1}, lambda x: x)
    )
    def test_rejects_bad_dtype(self, bad_dtype):
        with pytest.raises(ValueError):
            sparse_identity(5,5,bad_dtype)



    def test_accuracy_1(self):
        out = sparse_identity(2,3,float)
        assert out == {0:{0:1,2:0}, 1:{1:1, 2:0}}
        assert type(out[0][0]) == float


    def test_accuracy_2(self):
        out = sparse_identity(3,2,int)
        assert out == {0:{0:1,1:0}, 1:{1:1}, 2:{1:0}}
        assert type(out[0][0]) == int


    def test_accuracy_3(self):
        out = sparse_identity(2,2, np.uint8)
        assert out == {0:{0:1,1:0}, 1:{1:1}}
        assert type(out[0][0]) == np.uint8


    def test_accuracy_4(self):
        out = sparse_identity(3,4, np.uint8)
        assert out == {0:{0:1,3:0}, 1:{1:1,3:0}, 2:{2:1,3:0}}
        assert type(out[0][0]) == np.uint8





