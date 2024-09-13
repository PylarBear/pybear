# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest
import numpy as np

from pybear.sparse_dict._linalg_validation import (
                                                    _dot_size_check,
                                                    _broadcast_check,
                                                    _matrix_shape_check,
                                                    _outer_len_check,
                                                    _inner_len_check
)

# PIZZA
from pybear.sparse_dict._linalg_validation import _symmetric_matmul_check


class TestDotSizeCheck:

    @pytest.mark.parametrize('non_dict',
         (None, np.pi, [1], {1}, min, 'junk', lambda x: x, True, 1)
    )
    def test_rejects_non_dict(self, non_dict):
        with pytest.raises(TypeError):
            _dot_size_check(non_dict, {0:{0:1,1:2}})

        with pytest.raises(TypeError):
            _dot_size_check({0: {0: 1, 1: 2}}, non_dict)

        with pytest.raises(TypeError):
            _dot_size_check(non_dict, non_dict)


    def test_accepts_vector_same_len(self):
        dict1 = {0: {0:1, 1:2, 2:0}}
        dict2 = {1: {1:2, 2:2}}
        _dot_size_check(dict1, dict2)


    def test_rejects_array(self):
        dict1 = {0: {0:1, 1:2, 2:0}, 1:{1:1,2:1}}
        dict2 = {1: {1:2, 2:2}, 2:{0:2,2:0}}
        with pytest.raises(ValueError):
            _dot_size_check(dict1, dict2)


    def test_rejects_diff_len(self):
        dict1 = {0: {0:1, 1:2}}
        dict2 = {0: {1:2, 2:2}}
        with pytest.raises(ValueError):
            _dot_size_check(dict1, dict2)


class TestBroadcastCheck:

    @pytest.mark.parametrize('non_dict',
         (None, np.pi, [1], {1,2}, min, 'junk', lambda x: x, True, 1)
    )
    def test_rejects_non_dict(self, non_dict):
        with pytest.raises(TypeError):
            _broadcast_check(non_dict, {0: {0: 1, 1: 2}})

        with pytest.raises(TypeError):
            _broadcast_check({0: {0: 1, 1: 2}}, non_dict)

        with pytest.raises(TypeError):
            _broadcast_check(non_dict, non_dict)

    @pytest.mark.parametrize('i', (1, 3))
    @pytest.mark.parametrize('j', (1, 3))
    @pytest.mark.parametrize('k', (2, 4))
    def test_accepts_good(self, i, j, k):
        dict1 = {_: {__: np.random.randint(1, 4) for __ in range(j)} for _ in range(i)}
        dict2 = {_: {__: np.random.randint(1, 4) for __ in range(k)} for _ in range(j)}
        _broadcast_check(dict1, dict2)

    @pytest.mark.parametrize('i', (2, 4))
    @pytest.mark.parametrize('j', (2, 4))
    @pytest.mark.parametrize('k', (3, 5))
    def test_rejects_bad(self, i, j, k):
        dict1 = {_: {__: np.random.randint(1, 4) for __ in range(i)} for _ in range(j)}
        dict2 = {_: {__: np.random.randint(1, 4) for __ in range(j)} for _ in range(k)}
        with pytest.raises(ValueError):
            _broadcast_check(dict1, dict2)


class TestMatrixShapeCheck:

    @pytest.mark.parametrize('non_dict',
         (None, np.pi, [1], {1}, min, 'junk', lambda x: x, True, 1)
    )
    def test_rejects_non_dict(self, non_dict):
        with pytest.raises(TypeError):
            _matrix_shape_check(non_dict)

    @pytest.mark.parametrize('i', (1, 2, 3))
    @pytest.mark.parametrize('j', (1, 2, 3))
    def test_accepts_good(self, i, j):
        dict1 = {_: {__: np.random.randint(1, 4) for __ in range(i)} for _ in range(j)}
        dict2 = {_: {__: np.random.randint(1, 4) for __ in range(i)} for _ in range(j)}
        _matrix_shape_check(dict1, dict2)

    @pytest.mark.parametrize('i', (1, 3, 5))
    @pytest.mark.parametrize('j', (2, 4, 6))
    def test_rejects_bad(self, i, j):
        dict1 = {_: {__: np.random.randint(1, 4) for __ in range(i)} for _ in range(i)}
        dict2 = {_: {__: np.random.randint(1, 4) for __ in range(j)} for _ in range(j)}
        with pytest.raises(ValueError):
            _matrix_shape_check(dict1, dict2)


class TestOuterLenCheck:

    @pytest.mark.parametrize('non_dict',
         (None, np.pi, [1], {1}, min, 'junk', lambda x: x, True, 1)
    )
    def test_rejects_non_dict(self, non_dict):
        with pytest.raises(TypeError):
            _outer_len_check(non_dict)

    @pytest.mark.parametrize('i', (1, 2, 3))
    def test_accepts_good(self, i):
        dict1 = {_: {__: np.random.randint(1, 4) for __ in range(3)} for _ in range(i)}
        dict2 = {_: {__: np.random.randint(1, 4) for __ in range(4)} for _ in range(i)}
        _outer_len_check(dict1, dict2)

    @pytest.mark.parametrize('i', (1, 3, 5))
    @pytest.mark.parametrize('j', (2, 4, 6))
    def test_rejects_bad(self, i, j):
        dict1 = {_: {__: np.random.randint(1, 4) for __ in range(3)} for _ in range(i)}
        dict2 = {_: {__: np.random.randint(1, 4) for __ in range(4)} for _ in range(j)}
        with pytest.raises(ValueError):
            _outer_len_check(dict1, dict2)


class TestInnerLenCheck:
    @pytest.mark.parametrize('non_dict',
        (None, np.pi, [1], {1}, min, 'junk', lambda x: x, True, 1)
    )
    def test_rejects_non_dict(self, non_dict):
        with pytest.raises(TypeError):
            _inner_len_check(non_dict)


    @pytest.mark.parametrize('i', (1, 2, 3))
    def test_accepts_good(self, i):
        dict1 = {_: {__: np.random.randint(1, 4) for __ in range(i)} for _ in range(3)}
        dict2 = {_: {__: np.random.randint(1, 4) for __ in range(i)} for _ in range(4)}
        _inner_len_check(dict1, dict2)


    @pytest.mark.parametrize('i', (1, 3, 5))
    @pytest.mark.parametrize('j', (2, 4, 6))
    def test_rejects_bad(self, i, j):
        dict1 = {_: {__: np.random.randint(1, 4) for __ in range(i)} for _ in range(3)}
        dict2 = {_: {__: np.random.randint(1, 4) for __ in range(j)} for _ in range(4)}
        with pytest.raises(ValueError):
            _inner_len_check(dict1, dict2)



class TestSymmetricMatmulCheck:

    @pytest.fixture
    def valid_dict1(self):
        return {0:{0:1,1:0}, 1:{1:1}}

    @pytest.fixture
    def valid_dict1_t(self):
        return {0:{0:1,1:0}, 1:{1:1}}

    @pytest.fixture
    def bad_dict1_t(self):
        return {0:{1:1}, 1:{0:1, 1:1}}

    @pytest.fixture
    def valid_dict2(self):
        return {0:{0:1, 1:2}, 1:{0:1, 1:2}}

    @pytest.fixture
    def valid_dict2_t(self):
        return {0:{0:1, 1:1}, 1:{0:2, 1:2}}

    @pytest.fixture
    def bad_dict2_t(self):
        return {0:{0:1, 1:2}, 1:{0:1, 1:2}}


    @pytest.mark.parametrize('non_dict',
         (None, np.pi, [1], {1}, min, 'junk', lambda x: x, True, 1)
    )
    def test_rejects_non_dict(self, non_dict, valid_dict1, valid_dict2):
        with pytest.raises(TypeError):
            _symmetric_matmul_check(non_dict, valid_dict2)
            _symmetric_matmul_check(valid_dict1, non_dict)
            _symmetric_matmul_check(non_dict, non_dict)


    def test_correctly_returns_true(self, valid_dict1, valid_dict1_t,
                                    valid_dict2, valid_dict2_t):
        assert _symmetric_matmul_check(valid_dict1, valid_dict1_t)
        assert _symmetric_matmul_check(valid_dict2, valid_dict2_t)


    def test_correctly_returns_false(self, valid_dict1, valid_dict2):
        assert not _symmetric_matmul_check(valid_dict2, valid_dict2)
        assert not _symmetric_matmul_check(valid_dict1, valid_dict2)


    def test_accepts_valid_dict1_transpose(self, valid_dict1,  valid_dict1_t):
        assert _symmetric_matmul_check(valid_dict1, valid_dict1_t,
                                           DICT1_TRANSPOSE=valid_dict1_t)


    def test_rejects_bad_dict1_transpose(self, valid_dict1, valid_dict2,
                                         bad_dict1_t):
        with pytest.raises(ValueError):
            _symmetric_matmul_check(valid_dict1, valid_dict2,
                                        DICT1_TRANSPOSE=bad_dict1_t)


    def test_accepts_valid_dict2_transpose(self, valid_dict1, valid_dict1_t):
        assert _symmetric_matmul_check(valid_dict1, valid_dict1_t,
                                           DICT2_TRANSPOSE=valid_dict1)


    def test_rejects_bad_dict2_transpose(self, valid_dict1, valid_dict2,
                                         bad_dict2_t):
        with pytest.raises(ValueError):
            _symmetric_matmul_check(valid_dict1, valid_dict2,
                                        DICT2_TRANSPOSE=bad_dict2_t)









