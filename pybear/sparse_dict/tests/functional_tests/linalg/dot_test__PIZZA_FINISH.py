# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest
import numpy as np

from pybear.sparse_dict._linalg import (
                                        core_dot,
                                        dot,
                                        core_hybrid_dot,
                                        hybrid_dot,
                                        core_gaussian_dot,
                                        gaussian_dot,
                                        core_symmetric_gaussian_dot

)





class TestCoreDot:

    # no _validation, only results

    def test_accuracy_1(self):
        assert core_dot({0:{0:1,1:2,2:0}}, {0:2,1:1,2:0}) == 4

    def test_accuracy_2(self):
        assert core_dot({0: 1, 1: 1, 2: 1}, {0:{0: 1, 1: 1, 2: 1}}) == 3

    def test_accuracy_3(self):
        assert core_dot({2:15}, {2:10}) == 150


class TestDot:

    @pytest.mark.parametrize('dict1',
        ('junk', -1, True, False, None, np.pi, lambda x:x, [1,2], (1,2), {1,2},
         min, int, {0:{'a':1}}, {0:{0:'a'}})
    )
    def test_rejects_non_dicts(self, dict1):
        with pytest.raises(TypeError):
            dot(dict1, {0: 0, 1: 1, 2: 0, 3: 1})

        with pytest.raises(TypeError):
            dot({0: 0, 1: 1, 2: 0, 3: 1}, dict1)

        with pytest.raises(TypeError):
            dot(dict1, dict1)


    @pytest.mark.parametrize('dict1',
        ({'a':1}, {np.nan:{np.nan,np.nan}}, {-1:{0:1}})
    )
    def test_rejects_bad_dicts(self, dict1):
        with pytest.raises(ValueError):
            dot(dict1, {0: 0, 1: 1, 2: 0, 3: 1})

        with pytest.raises(ValueError):
            dot({0: 0, 1: 1, 2: 0, 3: 1}, dict1)

        with pytest.raises(ValueError):
            dot(dict1, dict1)


    def test_rejects_non_vector(self):
        with pytest.raises(ValueError):
            dot({0: {2: 1}, 1: {0: 1, 2: 0}}, {0: 1, 1: 1, 2: 1})

        with pytest.raises(ValueError):
            dot({0: 1, 1: 1, 2: 1}, {0: {2: 1}, 1: {0: 1, 2: 0}})

        with pytest.raises(ValueError):
            dot({0: {2: 1}, 1: {0: 1, 2: 0}}, {0: {2: 1}, 1: {0: 1, 2: 0}})


    def test_rejects_diff_len_vectors(self):

        with pytest.raises(ValueError):
            dot({0: 1, 1: 1, 3: 1}, {0: 1, 1: 1, 2: 1})

        with pytest.raises(ValueError):
            dot({0: {3: 1}}, {0: 1, 1: 1, 2: 1})

        with pytest.raises(ValueError):
            dot({0: 1, 1: 1, 3: 1}, {0: {2: 1}, 1: {0: 1, 2: 0}})

        with pytest.raises(ValueError):
            dot({0: {3: 1}}, {0: {2: 1}})


    def test_accuracy_1(self):
        assert dot({0:{0:1,1:2,2:0}}, {0:2,1:1,2:0}) == 4

    def test_accuracy_2(self):
        assert dot({0: 1, 1: 1, 2: 1}, {0:{0: 1, 1: 1, 2: 1}}) == 3

    def test_accuracy_3(self):
        assert dot({2:15}, {2:10}) == 150


class TestCoreHybridDot:

    # no _validation, only results

    def test_accuracy_1(self):
        assert core_hybrid_dot({0:{0:1,1:2,2:0}}, {0:2,1:1,2:0}) == 4

    def test_accuracy_2(self):
        assert core_hybrid_dot({0: 1, 1: 1, 2: 1}, {0:{0: 1, 1: 1, 2: 1}}) == 3

    def test_accuracy_3(self):
        assert core_hybrid_dot({2:15}, {2:10}) == 150


class TestHybridDot:

    @pytest.mark.parametrize('dict1',
        ('junk', -1, True, False, None, np.pi, lambda x:x, [1,2], (1,2), {1,2},
         min, int, {0:{'a':1}}, {0:{0:'a'}})
    )
    def test_rejects_non_dicts(self, dict1):
        with pytest.raises(TypeError):
            hybrid_dot(dict1, {0: 0, 1: 1, 2: 0, 3: 1})

        with pytest.raises(TypeError):
            hybrid_dot({0: 0, 1: 1, 2: 0, 3: 1}, dict1)

        with pytest.raises(TypeError):
            hybrid_dot(dict1, dict1)


    @pytest.mark.parametrize('dict1',
        ({'a':1}, {np.nan:{np.nan,np.nan}}, {-1:{0:1}})
    )
    def test_rejects_bad_dicts(self, dict1):
        with pytest.raises(ValueError):
            hybrid_dot(dict1, {0: 0, 1: 1, 2: 0, 3: 1})

        with pytest.raises(ValueError):
            hybrid_dot({0: 0, 1: 1, 2: 0, 3: 1}, dict1)

        with pytest.raises(ValueError):
            hybrid_dot(dict1, dict1)


    def test_rejects_non_vector(self):
        with pytest.raises(ValueError):
            hybrid_dot({0: {2: 1}, 1: {0: 1, 2: 0}}, {0: 1, 1: 1, 2: 1})

        with pytest.raises(ValueError):
            hybrid_dot({0: 1, 1: 1, 2: 1}, {0: {2: 1}, 1: {0: 1, 2: 0}})

        with pytest.raises(ValueError):
            hybrid_dot({0: {2: 1}, 1: {0: 1, 2: 0}}, {0: {2: 1}, 1: {0: 1, 2: 0}})


    def test_rejects_diff_len_vectors(self):

        with pytest.raises(ValueError):
            hybrid_dot({0: 1, 1: 1, 3: 1}, {0: 1, 1: 1, 2: 1})

        with pytest.raises(ValueError):
            hybrid_dot({0: {3: 1}}, {0: 1, 1: 1, 2: 1})

        with pytest.raises(ValueError):
            hybrid_dot({0: 1, 1: 1, 3: 1}, {0: {2: 1}, 1: {0: 1, 2: 0}})

        with pytest.raises(ValueError):
            hybrid_dot({0: {3: 1}}, {0: {2: 1}})


    def test_accuracy_1(self):
        assert hybrid_dot({0:{0:1,1:2,2:0}}, {0:2,1:1,2:0}) == 4

    def test_accuracy_2(self):
        assert hybrid_dot({0: 1, 1: 1, 2: 1}, {0:{0: 1, 1: 1, 2: 1}}) == 3

    def test_accuracy_3(self):
        assert hybrid_dot({2:15}, {2:10}) == 150


"""


def core_gaussian_dot(DICT1, DICT2, sigma, return_as='SPARSE_DICT'):
    '''Gaussian dot product.  [] of DICT1 are dotted with [] from DICT2.  
    There is no protection here to prevent
        dissimilar sized inner dicts from dotting.'''


def gaussian_dot(DICT1, DICT2, sigma, return_as='SPARSE_DICT'):

    DICT1 = val._dict_init(DICT1)
    DICT2 = val._dict_init(DICT2)
    val._inner_len_check(DICT1, DICT2)

    GAUSSIAN_DOT = core_gaussian_dot(DICT1, DICT2, sigma, return_as=return_as)

    return GAUSSIAN_DOT


def core_symmetric_gaussian_dot(DICT1, sigma, return_as='SPARSE_DICT'):
    '''Gaussian dot product for a symmetric result.  [] of DICT1 are dotted 
    with [] from DICT2.  There is no protection
        here to prevent dissimilar sized inner dicts from dotting.'''

    UNZIP1 = unzip_to_ndarray_float64(DICT1)[0]

    final_inner_len = len(UNZIP1)

    GAUSSIAN_DOT = np.zeros((final_inner_len, final_inner_len),
                            dtype=np.float64)

    for outer_key1 in range(len(UNZIP1)):
        for outer_key2 in range(outer_key1 + 1):  # HAVE TO GET DIAGONAL SO +1
            gaussian_dot = np.sum(
                (UNZIP1[outer_key1] - UNZIP1[outer_key2]) ** 2)
            GAUSSIAN_DOT[outer_key1][outer_key2] = gaussian_dot
            GAUSSIAN_DOT[outer_key2][outer_key1] = gaussian_dot

    del UNZIP1

    GAUSSIAN_DOT = np.exp(-GAUSSIAN_DOT / (2 * sigma ** 2))

    if return_as == 'SPARSE_DICT':
        GAUSSIAN_DOT = zip_array(GAUSSIAN_DOT, dtype=float)

    return GAUSSIAN_DOT


"""











