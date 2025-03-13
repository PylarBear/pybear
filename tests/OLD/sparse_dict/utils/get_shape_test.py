# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest
import numpy as np
from pybear.sparse_dict import get_shape



@pytest.fixture
def good_sd_1():
    return {
            0: {0: 1, 2: 1},
            1: {1: 3, 2: 0},
            2: {0: 2, 2: 2},
            3: {0: 1, 1: 3, 2: 0}
    }

@pytest.fixture
def good_sd_2():
    return {0: {0:1,1:2,2:3,3:4}}


@pytest.fixture
def bad_sd_1():
    return {
            0: {0: 1, 1: 1},
            1: {1: 3, 1: 0, 2: 1},
            2: {0: 0, 1: 0, 2: 2},
            3: {0: 1, 1: 3}
    }


@pytest.fixture
def bad_sd_2():
    return {0:1, 1:2, 2:3}


@pytest.fixture
def good_np_1():
    return np.random.randint(0,10,(11,13))


@pytest.fixture
def good_np_2():
    return np.random.randint(0,10,5)


class TestGetShape:

    @pytest.mark.parametrize('name', ('tests', 'name', '123', ']0#)48#'))
    def test_accepts_str_name(self, name, good_sd_1):
        get_shape(name, good_sd_1, 'row')


    @pytest.mark.parametrize('name', (0, np.pi, [1], None, True, lambda x: x))
    def test_rejects_non_str_name(self, name, good_sd_1):
        with pytest.raises(TypeError):
            get_shape(name, good_sd_1, 'row')


    @pytest.mark.parametrize('orient', (0, np.pi, [1], None, True, lambda x: x))
    def test_rejects_non_str_orient(self, orient, good_sd_1):
        with pytest.raises(TypeError):
            get_shape('name', good_sd_1, orient)


    @pytest.mark.parametrize('orient', ('tests', 'name', '123', ']0#)48#'))
    def test_rejects_bad_str_orient(self, orient, good_sd_1):
        with pytest.raises(ValueError):
            get_shape('name', good_sd_1, orient)


    @pytest.mark.parametrize('orient', ('row', 'column'))
    def test_accepts_good_str_orient(self, orient, good_sd_1):
        get_shape('name', good_sd_1, orient)


    @pytest.mark.parametrize('junk_object',
        (0, np.pi, True, None, min, {'a':1}, lambda x: x, 'more junk')
    )
    def test_rejects_junk_objects(self, junk_object):

        if junk_object is None:
            pytest.xfail(reason = f'None is handled in the np way')

        with pytest.raises(TypeError):
            get_shape('name', junk_object, 'row')


    def test_rejects_bad_sd(self, bad_sd_1, bad_sd_2):
        with pytest.raises(ValueError):
            get_shape('test_sd', bad_sd_1, given_orientation='ROW')
            get_shape('test_sd', bad_sd_2, given_orientation='ROW')


    def test_accepts_np(self, good_np_1):
        get_shape('name', good_np_1, 'row')


    def test_rejects_ragged(self):
        with pytest.raises(TypeError):
            get_shape('name', [[0,1,2],[2,3],[4,5,6]], 'row')



    NAMES = ['None'] + \
            [f"{x}{_}" for x in ['NP_ARRAY', 'SPARSE_DICT'] for _ in range(1,6)]

    OBJECTS = (
                None,
                [],
                [[], []],
                [2, 3, 4],
                [[2, 3, 4]],
                [[1, 2], [3, 4]],
                {},
                {0: {}, 1: {}},
                {0: 2, 1: 3, 2: 4},
                {0: {0: 2, 1: 3, 2: 4}},
                {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}
    )

    ROW_KEYS = ((), (0,), (2, 0), (3,), (1, 3), (2, 2),
           (0,), (2, 0), (3,), (1, 3), (2, 2)
    )

    COLUMN_KEYS = ((), (0,), (0, 2), (3,), (3, 1), (2, 2),
           (0,), (0, 2), (3,), (3, 1), (2, 2)
    )

    @pytest.mark.parametrize('name, object, key', zip(NAMES, OBJECTS, ROW_KEYS))
    def test_accuracy_row(self, name, object, key):
        assert get_shape(name, object, 'ROW') == key


    @pytest.mark.parametrize('name, object, key', zip(NAMES, OBJECTS, COLUMN_KEYS))
    def test_accuracy_column(self, name, object, key):
        assert get_shape(name, object, 'COLUMN') == key

























