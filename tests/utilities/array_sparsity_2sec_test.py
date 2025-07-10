# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause



import pytest

import numpy as np


from pybear.utilities._array_sparsity import array_sparsity



@pytest.fixture(scope='module')
def _shape_1D():
    return (10, )


@pytest.fixture(scope='module')
def _base_X_1D():
    return (10, )


@pytest.fixture(scope='module')
def _shape_2D():
    return (10, 10)


@pytest.fixture(scope='module')
def _shape_2D():
    return (10, 10)



class TestValidation:
    # TypeError for not an array-like
    # TypeError if cant be converted by list()
    # ValueError if size==0
    @pytest.mark.parametrize('_format', ('np', 'pd', 'pl'))
    @pytest.mark.parametrize('_dim', (1, 2))
    def test_verify_accepts_np_pd(self, _format, _dim):

        # if _format == 'np':
        #
        # elif _format == 'pd':
        #
        # else:
        #     raise Exception

        # 1D
        _base_X = np.random.randint(0, 10, (10,))
        array_sparsity(_base_X)

        # 2D
        array_sparsity(np.random.randint(0, 10, (37, 13)))


    # raises TypeError for not an array-like
    @pytest.mark.parametrize('a', ('junk', None, {'a':1, 'b':2}))
    def test_not_an_array_like(self, a):
        with pytest.raises(TypeError):
            array_sparsity(a)


    # raises TypeError if cant be converted by list()
    @pytest.mark.parametrize('a', (3, np.pi, float('inf')))
    def test_cannot_be_converted_by_py_list(self, a):
        with pytest.raises(TypeError):
            array_sparsity(a)


    # raises ValueError for empty array-like
    def test_empty_array(self):

        with pytest.raises(ValueError):
            array_sparsity(np.array([]))


class TestAccuracy1D:


    @staticmethod
    @pytest.fixture
    def good_array():
        return np.random.randint(1, 10, (10, ), dtype=np.uint8)


    def test_1(self, good_array):
        assert int(array_sparsity(good_array)) == 0

    def test_2(self, good_array):
        good_array[0] = 0
        assert int(array_sparsity(good_array)) == 10

    def test_3(self, good_array):
        good_array[:] = 0
        assert int(array_sparsity(good_array)) == 100

    def test_4(self, good_array):
        good_array[:5] = 0
        assert int(array_sparsity(good_array)) == 50


class TestAccuracy2D:


    @staticmethod
    @pytest.fixture
    def good_array():
        return np.random.randint(1, 10, (10, 10), dtype=np.uint8)


    def test_1(self, good_array):
        assert int(array_sparsity(good_array)) == 0

    def test_2(self, good_array):
        good_array[0, :] = 0
        assert int(array_sparsity(good_array)) == 10

    def test_3(self, good_array):
        good_array[:, :] = 0
        assert int(array_sparsity(good_array)) == 100

    def test_4(self, good_array):
        good_array[0, 0] = 0
        assert int(array_sparsity(good_array)) == 1

    def test_5(self, good_array):
        for i in range(10):
            good_array[i, i] = 0
        assert int(array_sparsity(good_array)) == 10





