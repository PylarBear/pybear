# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest
import numpy as np

from pybear.preprocessing.MinCountTransformer._transform._tcbc_update \
    import _tcbc_update


# _tcbc_update(
#     old_tcbc: TotalCountsByColumnType,
#     recursion_tcbc: TotalCountsByColumnType,
#     MAP_DICT: dict[int: int]
# )


class TestTCBCUpdate:


    @staticmethod
    @pytest.fixture
    def good_self_tcbc():
        return {
            0: {'a': 25, 'b': 50, 'c': 10, np.nan: 15},
            1: {0: 73, 1: 21, np.nan: 6},
            2: {0: 11, 1: 23, 2: 38, 3: 11, np.nan: 17}
        }


    @staticmethod
    @pytest.fixture
    def good_recursion_tcbc():
        return {
            0: {'a': 15, 'b': 40, 'c': 5, np.nan: 10},
            1: {0: 53, 1: 16, np.nan: 1},
            2: {0: 5, 1: 17, 2: 30, 3: 6, np.nan: 12}
        }


    @staticmethod
    @pytest.fixture
    def good_recursion_tcbc_2():
        return {
            0: {'a': 25, 'b': 50, 'c': 10, np.nan: 15},
            1: {0: 73, 1: 21, np.nan: 6},
            2: {0: 11, 1: 23, 2: 38, 3: 11, np.nan: 17}
        }


    @staticmethod
    @pytest.fixture
    def bad_recursion_tcbc():
        return {
            0: {'a': 35, 'b': 40, 'c': 5, np.nan: 10},
            1: {0: 73, 1: 16, np.nan: 1},
            2: {0: 25, 1: 17, 2: 30, 3: 6, np.nan: 12}
        }


    @staticmethod
    @pytest.fixture
    def map_dict():
        return {0: 0, 1: 1, 2: 2}


    def test_except_on_bad_ct(self, good_self_tcbc, bad_recursion_tcbc,
                              map_dict):
        # a deeper recursions unq ct > higher recursions ct
        with pytest.raises(AssertionError):

            _tcbc_update(
                good_self_tcbc,
                bad_recursion_tcbc,
                map_dict
            )



    def test_accuracy_1(self, good_self_tcbc, good_recursion_tcbc, map_dict):

        out = _tcbc_update(
            good_self_tcbc,
            good_recursion_tcbc,
            map_dict
        )

        assert out == {
                0: {'a': 15, 'b': 40, 'c': 5, np.nan: 10},
                1: {0: 53, 1: 16, np.nan: 1},
                2: {0: 5, 1: 17, 2: 30, 3: 6, np.nan: 12}
            }


    def test_accuracy_2(self, good_self_tcbc, good_recursion_tcbc_2, map_dict):

        out = _tcbc_update(
            good_self_tcbc,
            good_recursion_tcbc_2,
            map_dict
        )

        assert out == good_self_tcbc





















