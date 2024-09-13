# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest
from pybear.sparse_dict._linalg import (
                                        core_sparse_transpose_brute_force,
                                        core_sparse_transpose_map
)
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



class TestCoreSparseTransposeBruteForce:

    def test_accuracy(self, start_sparse_dict_1, end_sparse_dict_1,
                      start_sparse_dict_2, end_sparse_dict_2,
                      start_sparse_dict_3, end_sparse_dict_3,
                      start_sparse_dict_4, end_sparse_dict_4
                      ):

        assert core_sparse_equiv(
                        core_sparse_transpose_brute_force(start_sparse_dict_1),
                        end_sparse_dict_1
        )

        assert core_sparse_equiv(
                        core_sparse_transpose_brute_force(start_sparse_dict_2),
                        end_sparse_dict_2
        )

        assert core_sparse_equiv(
                        core_sparse_transpose_brute_force(start_sparse_dict_3),
                        end_sparse_dict_3
        )

        assert core_sparse_equiv(
                        core_sparse_transpose_brute_force(start_sparse_dict_4),
                        end_sparse_dict_4
        )

    def test_accuracy_empty(self):

        assert core_sparse_equiv(
                                    core_sparse_transpose_brute_force({}),
                                    {}
        )

        assert core_sparse_equiv(
                                    core_sparse_transpose_brute_force({0: {}}),
                                    {0: {}}
        )


class TestCoreSparseTransposeMap:

    def test_accuracy(self, start_sparse_dict_1, end_sparse_dict_1,
                      start_sparse_dict_2, end_sparse_dict_2,
                      start_sparse_dict_3, end_sparse_dict_3,
                      start_sparse_dict_4, end_sparse_dict_4
                      ):

        assert core_sparse_equiv(
                                core_sparse_transpose_map(start_sparse_dict_1),
                                end_sparse_dict_1
        )

        assert core_sparse_equiv(
                                core_sparse_transpose_map(start_sparse_dict_2),
                                end_sparse_dict_2
        )

        assert core_sparse_equiv(
                                core_sparse_transpose_map(start_sparse_dict_3),
                                end_sparse_dict_3
        )

        assert core_sparse_equiv(
                                core_sparse_transpose_map(start_sparse_dict_4),
                                end_sparse_dict_4
        )


    def test_accuracy_empty(self):

        assert core_sparse_equiv(
                                    core_sparse_transpose_map({}),
                                    {}
        )

        assert core_sparse_equiv(
                                    core_sparse_transpose_map({0: {}}),
                                    {0: {}}
        )








