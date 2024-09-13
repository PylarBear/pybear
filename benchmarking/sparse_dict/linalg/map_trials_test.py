# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest
from pybear.sparse_dict.tests.benchmarking.linalg.core_transpose_map_trials import (
                core_sparse_transpose_map_duality,
                core_sparse_transpose_map_no_duality_1,
                core_sparse_transpose_map_no_duality_2
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



class TestMapDuality:

    def test_accuracy(self, start_sparse_dict_1, end_sparse_dict_1,
                      start_sparse_dict_2, end_sparse_dict_2,
                      start_sparse_dict_3, end_sparse_dict_3,
                      start_sparse_dict_4, end_sparse_dict_4
                      ):

        assert core_sparse_equiv(
                        core_sparse_transpose_map_duality(start_sparse_dict_1),
                        end_sparse_dict_1
        )

        assert core_sparse_equiv(
                        core_sparse_transpose_map_duality(start_sparse_dict_2),
                        end_sparse_dict_2
        )

        assert core_sparse_equiv(
                        core_sparse_transpose_map_duality(start_sparse_dict_3),
                        end_sparse_dict_3
        )

        assert core_sparse_equiv(
                        core_sparse_transpose_map_duality(start_sparse_dict_4),
                        end_sparse_dict_4
        )


class TestMapNoDuality1:

    def test_accuracy(self, start_sparse_dict_1, end_sparse_dict_1,
                      start_sparse_dict_2, end_sparse_dict_2,
                      start_sparse_dict_3, end_sparse_dict_3,
                      start_sparse_dict_4, end_sparse_dict_4
                      ):

        assert core_sparse_equiv(
                        core_sparse_transpose_map_no_duality_1(start_sparse_dict_1),
                        end_sparse_dict_1
        )

        assert core_sparse_equiv(
                        core_sparse_transpose_map_no_duality_1(start_sparse_dict_2),
                        end_sparse_dict_2
        )

        assert core_sparse_equiv(
                        core_sparse_transpose_map_no_duality_1(start_sparse_dict_3),
                        end_sparse_dict_3
        )

        assert core_sparse_equiv(
                        core_sparse_transpose_map_no_duality_1(start_sparse_dict_4),
                        end_sparse_dict_4
        )


class TestMapNoDuality2:

    def test_accuracy(self, start_sparse_dict_1, end_sparse_dict_1,
                      start_sparse_dict_2, end_sparse_dict_2,
                      start_sparse_dict_3, end_sparse_dict_3,
                      start_sparse_dict_4, end_sparse_dict_4
                      ):

        assert core_sparse_equiv(
                core_sparse_transpose_map_no_duality_2(start_sparse_dict_1),
                end_sparse_dict_1
        )

        assert core_sparse_equiv(
                core_sparse_transpose_map_no_duality_2(start_sparse_dict_2),
                end_sparse_dict_2
        )

        assert core_sparse_equiv(
                core_sparse_transpose_map_no_duality_2(start_sparse_dict_3),
                end_sparse_dict_3
        )

        assert core_sparse_equiv(
                core_sparse_transpose_map_no_duality_2(start_sparse_dict_4),
                end_sparse_dict_4
        )


