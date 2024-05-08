# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

from pybear.sparse_dict._random_ import randint
from pybear.sparse_dict._utils import outer_len



@pytest.fixture
def randint_sd_dense():
    return randint(0,10,(5,5),0,int)


@pytest.fixture
def randint_sd_sparse():
    return randint(0,10,(5,5),90,int)




class TestAcceptsRandint:

    def test_accepts_dense(self, randint_sd_dense):

        assert outer_len(randint_sd_dense) == 5


    def test_accepts_sparse(self, randint_sd_sparse):

        assert outer_len(randint_sd_sparse) == 5






