# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest
import numpy as np
from pybear.sparse_dict._linalg import core_sparse_transpose
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
         (0, True, None, np.pi, [1], (1,), {1,2}, lambda x: x, 'junk', min)
    )
    def test_rejects_non_dict(self, non_dict):
        with pytest.raises(TypeError):
            _dot_size_check(non_dict, {0:{0:1}})

        with pytest.raises(TypeError):
            _dot_size_check({0:{0:1}}, non_dict)

        with pytest.raises(TypeError):
            _dot_size_check(non_dict, non_dict)


    @pytest.mark.parametrize('bad_dict',
         ({0:{'a':1}}, {0: {0:'a'}}, {'a': {0:1}}, {0:{np.nan:np.nan}})
    )
    def test_rejects_bad_dict(self, bad_dict):
        with pytest.raises(TypeError):
            _dot_size_check(bad_dict, {0:{0:1}})

        with pytest.raises(TypeError):
            _dot_size_check({0:{0:1}}, bad_dict)

        with pytest.raises(TypeError):
            _dot_size_check(bad_dict, bad_dict)


    @pytest.fixture
    def sd_vector(self):
        return {0:{0:1,1:2}}


    def test_accepts_outer_vectors(self, sd_vector):
        _dot_size_check(sd_vector, sd_vector)


    def test_accepts_inner_vectors(self, sd_vector):
        _dot_size_check(sd_vector[0], sd_vector[0])


    @pytest.mark.parametrize('sd1, sd2', (({0: {}}, {0: {}}), ({}, {})))
    def test_accepts_empty(self, sd1, sd2):
        _dot_size_check(sd1, sd2)


    def test_rejects_arrays(self, sd_vector):
        sd_array = {0:{0:1,1:2}, 1:{0:2,1:1}}

        with pytest.raises(ValueError):
            _dot_size_check(sd_vector, sd_array)

        with pytest.raises(ValueError):
            _dot_size_check(sd_array, sd_vector)

        with pytest.raises(ValueError):
            _dot_size_check(sd_array, sd_array)


    def test_passes_equal_sized_vectors(self, sd_vector):

        _dot_size_check(sd_vector, {0:{1:2}})
        _dot_size_check({0:{1:1}}, sd_vector)


    def test_fails_diff_sized_vectors(self, sd_vector):

        with pytest.raises(ValueError):
            _dot_size_check(sd_vector, {0:{1000:2}})
        with pytest.raises(ValueError):
            _dot_size_check({0:{3:1}}, sd_vector)



class TestBroadcastCheck:

    @pytest.mark.parametrize('non_dict',
         (0, True, None, np.pi, [1], (1,), {1,2}, lambda x: x, 'junk', min)
    )
    def test_rejects_non_dict(self, non_dict):
        with pytest.raises(TypeError):
            _broadcast_check(non_dict, {0:{0:1}})

        with pytest.raises(TypeError):
            _broadcast_check({0:{0:1}}, non_dict)

        with pytest.raises(TypeError):
            _broadcast_check(non_dict, non_dict)


    @pytest.mark.parametrize('bad_dict',
         ({0:{'a':1}}, {0: {0:'a'}}, {'a': {0:1}}, {0:{np.nan:np.nan}})
    )
    def test_rejects_bad_dict(self, bad_dict):
        with pytest.raises(TypeError):
            _broadcast_check(bad_dict, {0:{0:1}})

        with pytest.raises(TypeError):
            _broadcast_check({0:{0:1}}, bad_dict)

        with pytest.raises(TypeError):
            _broadcast_check(bad_dict, bad_dict)


    def test_accepts_inner_dicts(self):

        outer_dict = {0:{0:1,1:2}}
        inner_dict = {0:1,1:2}

        _broadcast_check(outer_dict, inner_dict)

        _broadcast_check(inner_dict, outer_dict)

        _broadcast_check(inner_dict, inner_dict)


    def test_accepts_outer_dicts(self):

        _broadcast_check({0:{0:1,1:2}}, {0:{0:1},1:{0:2}})


    @pytest.mark.parametrize('sd1, sd2',
                             (({0: {}}, {0: {}}),
                             ({}, {}))
    )
    def test_accepts_empty(self, sd1, sd2):
        _broadcast_check(sd1, sd2)


    @pytest.fixture
    def good_array_1(self):
        return {0:{0:1,1:2,2:3},1:{0:2,1:1,2:0}}


    @pytest.fixture
    def good_array_2(self):
        return {0:{0:1,1:2},1:{0:2,1:1},2:{0:2,1:2}}


    def test_passes_correctly_shaped_arrays(self, good_array_1, good_array_2):
        _broadcast_check(good_array_1, good_array_2)


    def test_fails_diff_shaped_arrays(self, good_array_1):
        with pytest.raises(ValueError):
            _broadcast_check(good_array_1, good_array_1)



class TestMatrixShapeCheck:

    @pytest.mark.parametrize('non_dict',
         (0, True, None, np.pi, [1], (1,), {1,2}, lambda x: x, 'junk', min)
    )
    def test_rejects_non_dict(self, non_dict):
        with pytest.raises(TypeError):
            _matrix_shape_check(non_dict, {0:{0:1}})

        with pytest.raises(TypeError):
            _matrix_shape_check({0:{0:1}}, non_dict)

        with pytest.raises(TypeError):
            _matrix_shape_check(non_dict, non_dict)


    @pytest.mark.parametrize('bad_dict',
         ({0:{'a':1}}, {0: {0:'a'}}, {'a': {0:1}}, {0:{np.nan:np.nan}}, {'a':'a'})
    )
    def test_rejects_bad_dict(self, bad_dict):
        with pytest.raises(TypeError):
            _matrix_shape_check(bad_dict, {0:{0:1}})

        with pytest.raises(TypeError):
            _matrix_shape_check({0:{0:1}}, bad_dict)

        with pytest.raises(TypeError):
            _matrix_shape_check(bad_dict, bad_dict)


    def test_accepts_inner_and_outer_dicts(self):

        _matrix_shape_check({0: {0: 1, 1: 2}}, {1: {1: 5}})
        _matrix_shape_check({0:1,1:2}, {1:10})


    @pytest.mark.parametrize('sd1, sd2',
                             (({0:{}}, {0:{}}), ({}, {}))
    )
    def test_accepts_empty(self, sd1, sd2):
        _broadcast_check(sd1, sd2)


    @pytest.mark.parametrize('array1, array2',
        (
        ({0:{0:1,1:2,2:3},1:{0:2,1:1,2:0}}, {0:{0:2,1:3,2:4},1:{0:5,1:6,2:7}}),
        ({0:1,1:2,2:3,3:4,4:5}, {0:9,1:8,2:7,3:6,4:5}),
        )
    )
    def test_passes_correctly_shaped_arrays(self, array1, array2):
        _matrix_shape_check(array1, array2)


    @pytest.mark.parametrize('array1, array2',
        (
        ({0:{0:1,1:2,2:3},1:{0:2,1:1,2:0}}, {0:{0:1,1:2,2:3,3:4,4:5}}),
        ({0:1,1:2,2:3,3:4,4:5}, {0:1,100:0})
        )
    )
    def test_fails_diff_shaped_arrays(self, array1, array2):
        with pytest.raises(ValueError):
            _matrix_shape_check(array1, array2)



class TestOuterLenCheck:

    @pytest.mark.parametrize('non_dict',
         (0, True, None, np.pi, [1], (1,), {1,2}, lambda x: x, 'junk', min)
    )
    def test_rejects_non_dict(self, non_dict):
        with pytest.raises(TypeError):
            _outer_len_check(non_dict, {0:{0:1}})

        with pytest.raises(TypeError):
            _outer_len_check({0:{0:1}}, non_dict)

        with pytest.raises(TypeError):
            _outer_len_check(non_dict, non_dict)


    @pytest.mark.parametrize('bad_dict',
         ({0:{'a':1}}, {0: {0:'a'}}, {'a': {0:1}}, {0:{np.nan:np.nan}})
    )
    def test_rejects_bad_dict(self, bad_dict):
        with pytest.raises(TypeError):
            _outer_len_check(bad_dict, {0:{0:1}})

        with pytest.raises(TypeError):
            _outer_len_check({0:{0:1}}, bad_dict)

        with pytest.raises(TypeError):
            _outer_len_check(bad_dict, bad_dict)


    @pytest.fixture
    def sd_vector(self):
        return {0:{0:1,1:2}}


    def test_accepts_outer_vectors(self, sd_vector):
        _outer_len_check(sd_vector, sd_vector)


    def test_rejects_inner_vectors(self, sd_vector):
        with pytest.raises(ValueError):
            _outer_len_check(sd_vector[0], sd_vector)

        with pytest.raises(ValueError):
            _outer_len_check(sd_vector, sd_vector[0])

        with pytest.raises(ValueError):
            _outer_len_check(sd_vector[0], sd_vector[0])


    @pytest.mark.parametrize('sd1, sd2',
                             (({0:{}}, {0:{0:1,1:2}}),
                             ({0:{0:1,1:2}}, {0:{}}),
                             ({0:{}}, {0:{}}))
    )
    def test_accepts_empty(self, sd1, sd2):
        _outer_len_check(sd1, sd2)


    def test_passes_equal_sized_arrays(self, sd_vector):

        _outer_len_check(sd_vector, {0:{1:2}})
        _outer_len_check({0:{1:1}}, sd_vector)


    def test_fails_diff_sized_arrays(self, sd_vector):

        with pytest.raises(ValueError):
            _outer_len_check(sd_vector, {0:{1000:2},1:{1000:0}})

        with pytest.raises(ValueError):
            _outer_len_check({0:{3:1},1:{0:1,3:0}}, sd_vector)


class TestInnerLenCheck:

    @pytest.mark.parametrize('non_dict',
         (0, True, None, np.pi, [1], (1,), {1,2}, lambda x: x, 'junk', min)
    )
    def test_rejects_non_dict(self, non_dict):
        with pytest.raises(TypeError):
            _inner_len_check(non_dict, {0:{0:1}})

        with pytest.raises(TypeError):
            _inner_len_check({0:{0:1}}, non_dict)

        with pytest.raises(TypeError):
            _inner_len_check(non_dict, non_dict)


    @pytest.mark.parametrize('bad_dict',
         ({0:{'a':1}}, {0: {0:'a'}}, {'a': {0:1}}, {0:{np.nan:np.nan}})
    )
    def test_rejects_bad_dict(self, bad_dict):
        with pytest.raises(TypeError):
            _inner_len_check(bad_dict, {0:{0:1}})

        with pytest.raises(TypeError):
            _inner_len_check({0:{0:1}}, bad_dict)

        with pytest.raises(TypeError):
            _inner_len_check(bad_dict, bad_dict)


    @pytest.fixture
    def sd_array(self):
        return {0:{0:1,100:2}}


    def test_accepts_outer_vectors(self, sd_array):
        _inner_len_check(sd_array, sd_array)


    def test_accepts_inner_vectors(self, sd_array):
        _inner_len_check(sd_array[0], sd_array)
        _inner_len_check(sd_array, sd_array[0])
        _inner_len_check(sd_array[0], sd_array[0])


    @pytest.mark.parametrize('sd1, sd2',
                             (({}, {}),
                             ({0:{}}, {0:{}}))
    )
    def test_accepts_empty(self, sd1, sd2):
        _inner_len_check(sd1, sd2)


    def test_passes_equal_sized_arrays(self, sd_array):

        _inner_len_check(sd_array, {0:{100:2}})
        _inner_len_check({0:{100:1}}, sd_array)


    def test_passes_equal_sized_vectors(self, sd_array):

        _inner_len_check(sd_array[0], {0:1,100:2})
        _inner_len_check({0:2,50:1,100:1}, sd_array[0])


    def test_fails_diff_sized_arrays(self, sd_array):

        with pytest.raises(ValueError):
            _inner_len_check(sd_array, {0:{1000:2},1:{1000:0}})

        with pytest.raises(ValueError):
            _inner_len_check({0:{3:1},1:{0:1,3:0}}, sd_array)


    def test_fails_diff_sized_vectors(self, sd_array):

        with pytest.raises(ValueError):
            _inner_len_check(sd_array[0], {0:1, 1000:0})

        with pytest.raises(ValueError):
            _inner_len_check({0:0}, sd_array[0])


class TestSymmetricMatmulCheck:

    @pytest.mark.parametrize('non_dict',
         (0, True, None, np.pi, [1], (1,), {1,2}, lambda x: x, 'junk', min)
    )
    def test_rejects_non_dict(self, non_dict):
        with pytest.raises(TypeError):
            _symmetric_matmul_check(non_dict, {0:{0:1}})

        with pytest.raises(TypeError):
            _symmetric_matmul_check({0:{0:1}}, non_dict)

        with pytest.raises(TypeError):
            _symmetric_matmul_check(non_dict, non_dict)


    @pytest.mark.parametrize('bad_dict',
         ({0:{'a':1}}, {0: {0:'a'}}, {'a': {0:1}}, {0:{np.nan:np.nan}})
    )
    def test_rejects_bad_dict(self, bad_dict):
        with pytest.raises(TypeError):
            _symmetric_matmul_check(bad_dict, {0:{0:1}})

        with pytest.raises(TypeError):
            _symmetric_matmul_check({0:{0:1}}, bad_dict)

        with pytest.raises(TypeError):
            _symmetric_matmul_check(bad_dict, bad_dict)


    @pytest.fixture
    def good_array(self):
        return {0:{0:2,1:0}, 1:{1:2}}


    @pytest.fixture
    def good_array2(self):
        return {0: {0:1, 1:3, 2:0}, 1: {0:3, 1:2, 2:1}}


    @pytest.fixture
    def bad_array(self):
        return ({0:{0:1,1:2}, 1:{2:3,3:4}})


    def test_accepts_outer_vectors(self, good_array):
        _symmetric_matmul_check(good_array, good_array)

    @pytest.mark.xfail(reason=f'pizza needs to to fix inner vector handling in '
                              f'core sparse transpose'
    )
    def test_accepts_inner_vectors(self, good_array):
        # with pytest.raises(ValueError):
        _symmetric_matmul_check(good_array[0], good_array)

        # with pytest.raises(ValueError):
        _symmetric_matmul_check(good_array, good_array[0])

        # with pytest.raises(ValueError):
        _symmetric_matmul_check(good_array[0], good_array[0])


    @pytest.mark.parametrize('sd1, sd2',
                             (({0:{}}, {0:{0:1,1:2}}),
                             ({0:{0:1,1:2}}, {0:{}}),
                             ({0:{}}, {0:{}}))
    )
    def test_accepts_empty(self, sd1, sd2):
        _symmetric_matmul_check(sd1, sd2)


    def test_fails_bad_dict1_transpose(self, good_array, bad_array):
        with pytest.raises(ValueError):
            _symmetric_matmul_check(
                                        good_array,
                                        good_array,
                                        DICT1_TRANSPOSE=bad_array
            )

    def test_fails_bad_dict2_transpose(self, good_array, bad_array):
        with pytest.raises(ValueError):
            _symmetric_matmul_check(
                                        good_array,
                                        good_array,
                                        DICT2_TRANSPOSE=bad_array
            )


    def test_fails_non_symmetric(self, good_array):

        assert not _symmetric_matmul_check(
                                            {0: {0: 1, 1: 2}, 1: {0: 2, 1: 4}},
                                            {0: {0: 5, 1: 0}, 1: {0: 1, 1: 0}}
            )

        assert not _symmetric_matmul_check(good_array, {0:{3:1},1:{0:1,3:0}})

        assert not _symmetric_matmul_check({0:{3:1},1:{0:1,3:0}}, good_array)


    def test_passes_symmetric(self, good_array, good_array2):

        assert _symmetric_matmul_check(
                                good_array,
                                core_sparse_transpose(good_array)
        )

        assert _symmetric_matmul_check(
                            good_array,
                            core_sparse_transpose(good_array),
                            DICT1_TRANSPOSE=core_sparse_transpose(good_array),
                            DICT2_TRANSPOSE=good_array,
        )


        assert _symmetric_matmul_check(
                                good_array2,
                                core_sparse_transpose(good_array2)
        )

        assert _symmetric_matmul_check(
                            good_array2,
                            core_sparse_transpose(good_array2),
                            DICT1_TRANSPOSE=core_sparse_transpose(good_array2),
                            DICT2_TRANSPOSE=good_array2,
        )




















