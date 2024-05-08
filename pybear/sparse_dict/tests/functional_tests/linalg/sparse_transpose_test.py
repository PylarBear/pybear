# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest
import numpy as np
from pybear.sparse_dict._linalg import sparse_transpose
from pybear.sparse_dict._utils import core_sparse_equiv


@pytest.fixture
def start_sparse_dict_1():
    return {
            0: {0:1, 1:2, 2:3},
            1: {0:4, 1:5, 2:6},
            2: {0:7, 1:8, 2:0}
    }

@pytest.fixture
def end_sparse_dict_1():
    return {
            0: {0:1, 1:4, 2:7},
            1: {0:2, 1:5, 2:8},
            2: {0:3, 1:6, 2:0}
    }


@pytest.fixture
def start_sparse_dict_2():
    return {0: {0:1, 1:2, 2:3, 3:4, 4:0}}

@pytest.fixture
def end_sparse_dict_2():
    return {0: {0:1}, 1: {0:2}, 2: {0: 3}, 3:{0: 4}, 4:{0:0}}


@pytest.fixture
def start_sparse_dict_3():
    return {0: {0:1}, 1: {0:2}, 2: {0: 3}, 3:{0: 4}, 4:{0:0}}

@pytest.fixture
def end_sparse_dict_3():
    return {0: {0:1, 1:2, 2:3, 3:4, 4:0}}



@pytest.fixture
def start_sparse_dict_4():
    return {0:1, 1:2, 2:3, 3:4, 4:0}

@pytest.fixture
def end_sparse_dict_4():
    return {0: {0:1}, 1: {0:2}, 2: {0: 3}, 3:{0: 4}, 4:{0:0}}



class TestSparseTranspose:


    @pytest.mark.parametrize('non_sd',
        (0, 1, np.nan, np.pi, [1,2], {1,2}, (1,2), True, False, None,
         min, lambda x: x, np.float64, 'junk', {np.nan, np.nan})
    )
    def test_rejects_non_sparse_dicts(self, non_sd):
        with pytest.raises(TypeError):
            sparse_transpose(non_sd)


    @pytest.mark.parametrize('bad_sd', ({'a':1}, {0: 'a'}))
    def test_rejects_bad_sparse_dicts(self, bad_sd):
        with pytest.raises(ValueError):
            sparse_transpose(bad_sd)


    def test_accepts_full_sparse_dict(self, start_sparse_dict_1):
        sparse_transpose(start_sparse_dict_1)


    def test_accepts_inner_sparse_dict(self, start_sparse_dict_4):
        sparse_transpose(start_sparse_dict_4)


    def test_accuracy(self, start_sparse_dict_1, end_sparse_dict_1,
                      start_sparse_dict_2, end_sparse_dict_2,
                      start_sparse_dict_3, end_sparse_dict_3,
                      start_sparse_dict_4, end_sparse_dict_4
                      ):

        assert core_sparse_equiv(
                                    sparse_transpose(start_sparse_dict_1),
                                    end_sparse_dict_1
        )

        assert core_sparse_equiv(
                                    sparse_transpose(start_sparse_dict_2),
                                    end_sparse_dict_2
        )

        assert core_sparse_equiv(
                                    sparse_transpose(start_sparse_dict_3),
                                    end_sparse_dict_3
        )

        assert core_sparse_equiv(
                                    sparse_transpose(start_sparse_dict_4),
                                    end_sparse_dict_4
        )


    def test_accuracy_empty(self):

        assert core_sparse_equiv(
                                    sparse_transpose({}),
                                    {}
        )

        assert core_sparse_equiv(
                                    sparse_transpose({0: {}}),
                                    {0: {}}
        )








