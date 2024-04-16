# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest
from pybear.sparse_dict._validation import _sparse_dict_check as sdc



@pytest.fixture
def good_sd():
    return {0:{0:1, 1:0}, 1:{1:1}}




def test_accepts_good_sd(good_sd):
    sdc(good_sd)


@pytest.mark.parametrize('x', ('junk', [], int, {1,2}, None, True, (), lambda: 0))
def test_rejects_non_dictionary(x):
    with pytest.raises(TypeError):
        sdc(x)


def test_rejects_empty_dictionary():
    with pytest.raises(ValueError):
        sdc({})


@pytest.mark.parametrize('x', ('junk', [], int, {1,2}, None, True, (), lambda: 0))
def test_rejects_non_dict_inner(x):
    with pytest.raises(TypeError):
        sdc({0:x, 1:{0:1,1:1}})


@pytest.mark.parametrize('x', (0.2432, True, False))
class RejectsNonIntegerAndBoolKeys:

    def test_outer_key(self, x):
        with pytest.raises(TypeError):
            sdc({x: {0:1, 1:0}, 1: {1:0}})

    def test_inner_key(self, x):
        with pytest.raises(TypeError):
            sdc({0: {x:1, 1:0}, 1: {1:0}})


@pytest.mark.parametrize('x', ('junk', [], int, {}, (), lambda: 0))
class TestRejectsNonNumbersAnywhere:

    def test_outer_key(self, x):
        with pytest.raises(TypeError):
            sdc({x: {1:1}, 1: {0:1, 1:1}})

    def test_inner_key(self, x):
        with pytest.raises(TypeError):
            sdc({0: {x:1}, 1: {0:1, 1:1}})

    def test_value(self, x):
        with pytest.raises(TypeError):
            sdc({0: {1:x}, 1: {0:1, 1:1}})


class TestValues:

    @pytest.mark.parametrize('x', (True, False))
    def test_accepts_bool_values(self, x):
        sdc({0:{0:1,1:1}, 1:{0:x, 1:0}})


    def test_accepts_np_nan_as_value(self):
        x = 1 # np.nan
        sdc({0:{0:x, 1:x}, 1:{0:x, 1:x}})


    def test_rejects_none_as_value(self):
        with pytest.raises(TypeError):
            sdc({0: {0: None, 1: None}, 1: {0: None, 1: None}})


class RejectsNoPlaceholders:

    def test_1(self):
        with pytest.raises(ValueError):
            sdc({0:{0:1,1:0}, 1:{0:1}})

    def test_2(self):
        with pytest.raises(ValueError):
            sdc({0:{0:1}, 1:{0:1, 1:0}})










