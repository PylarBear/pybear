# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest
import numpy as np
from pybear.sparse_dict._utils import (
                                        outer_len,
                                        inner_len,
                                        inner_len_quick,
                                        shape_,
                                        size_,
                                        clean,
                                        sparsity,
                                        core_sparse_equiv,
                                        safe_sparse_equiv
)


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
            1: {0: 3, 1: 0, 2: 1},
            2: {0: 0, 1: 0, 2: 2},
            3: {0: 1, 1: 3}
    }


@pytest.fixture
def bad_sd_2():
    return {0:1, 1:2, 2:3}


@pytest.fixture
def errant_list():
    return np.random.randint(5)





class TestOuterLen:
    @pytest.mark.parametrize('junk_sd',
        ('junk', 1, [1, 2, 3], {0, 1, 2}, np.pi, None, True, min, np.ndarray)
    )
    def test_rejects_non_sd(self, junk_sd):
        with pytest.raises(TypeError):
            outer_len(junk_sd)


    def test_rejects_bad_sd(self, bad_sd_1, bad_sd_2):
        with pytest.raises(ValueError):
            outer_len(bad_sd_1)
            outer_len(bad_sd_2)


    def test_accuracy(self, good_sd_1, good_sd_2):
        assert outer_len(good_sd_1) == 4
        assert outer_len(good_sd_2) == 1


class TestInnerLenQuick:

    @pytest.mark.parametrize('junk_sd',
        ('junk', 1, [1, 2, 3], {0, 1, 2}, np.pi, None, True, min, np.ndarray)
    )
    def test_rejects_non_sd(self, junk_sd):
        with pytest.raises(TypeError):
            inner_len_quick(junk_sd)

    @pytest.mark.xfail(reason='does not have validation to handle bad dict')
    def test_rejects_bad_sd1(self, bad_sd_1):
        with pytest.raises(ValueError):
            inner_len_quick(bad_sd_1)

    @pytest.mark.xfail(reason='does not have validation to handle inner dict')
    def test_rejects_bad_sd2(self, bad_sd_2):
        with pytest.raises(ValueError):
            inner_len_quick(bad_sd_2)


    def test_accuracy(self, good_sd_1, good_sd_2):
        assert inner_len_quick(good_sd_1) == 3
        assert inner_len_quick(good_sd_2) == 4


class TestInnerLen:
    @pytest.mark.parametrize('junk_sd',
        ('junk', 1, [1, 2, 3], {0, 1, 2}, np.pi, None, True, min, np.ndarray)
    )
    def test_rejects_non_sd(self, junk_sd):
        with pytest.raises(TypeError):
            inner_len(junk_sd)


    def test_rejects_bad_sd(self, bad_sd_1, bad_sd_2):
        with pytest.raises(ValueError):
            inner_len(bad_sd_1)
            inner_len(bad_sd_2)


    def test_accuracy(self, good_sd_1, good_sd_2):
        assert inner_len(good_sd_1) == 3
        assert inner_len(good_sd_2) == 4


class TestSize:

    @pytest.mark.parametrize('junk_sd',
        ('junk', 1, [1, 2, 3], {0, 1, 2}, np.pi, None, True, min, np.ndarray)
    )
    def test_rejects_non_sd(self, junk_sd):
        with pytest.raises(TypeError):
            size_(junk_sd)


    def test_rejects_bad_sd(self, bad_sd_1, bad_sd_2):
        with pytest.raises(ValueError):
            size_(bad_sd_1)
            size_(bad_sd_2)


    def test_accuracy(self, good_sd_1, good_sd_2):
        assert size_(good_sd_1) == 12
        assert size_(good_sd_2) == 4


class TestShape:

    @pytest.mark.parametrize('junk_sd',
        ('junk', 1, [1, 2, 3], {0, 1, 2}, np.pi, None, True, min, np.ndarray)
    )
    def test_rejects_non_sd(self, junk_sd):
        if junk_sd is None:
            pytest.xfail(reason=f'has numpy-like handling for None')

        with pytest.raises(TypeError):
            shape_(junk_sd)


    def test_rejects_bad_sd(self, bad_sd_1, bad_sd_2):
        with pytest.raises(ValueError):
            shape_(bad_sd_1)
            shape_(bad_sd_2)


    def test_accuracy(self, good_sd_1, good_sd_2):
        assert shape_(good_sd_1) == (4, 3)
        assert shape_(good_sd_2) == (1, 4)


class TestClean:
    @pytest.mark.parametrize('junk_sd',
        ('junk', 1, [1, 2, 3], {0, 1, 2}, np.pi, None, True, min, np.ndarray)
    )
    def test_rejects_non_sd(self, junk_sd):
            with pytest.raises(TypeError):
                clean(junk_sd)


    def test_returns_an_already_clean_sd(self, good_sd_1, good_sd_2):
        assert clean(good_sd_1) == good_sd_1
        assert clean(good_sd_2) == good_sd_2

    def test_cleans_a_messed_up_sd(self, bad_sd_1):

        cleaned_sd = {
                        0: {0: 1, 1: 1, 2: 0},
                        1: {0: 3, 2: 1},
                        2: {2: 2},
                        3: {0: 1, 1: 3, 2: 0}
        }

        assert clean(bad_sd_1) == cleaned_sd


class TestSparsity:

    @pytest.mark.parametrize('junk_sd',
        ('junk', 1, [1, 2, 3], {0, 1, 2}, np.pi, None, True, min, np.ndarray)
    )
    def test_rejects_non_sd(self, junk_sd):
            with pytest.raises(TypeError):
                sparsity(junk_sd)


    def test_rejects_bad_sd(self, bad_sd_1, bad_sd_2):
        with pytest.raises(ValueError):
            sparsity(bad_sd_1)
            sparsity(bad_sd_2)


    def test_accuracy(self, good_sd_1, good_sd_2):
        assert round(sparsity(good_sd_1), 1) == 41.7
        assert sparsity(good_sd_2) == 0
        assert sparsity({0:{10:0}}) == 100


def core_sparse_equiv(DICT1, DICT2):
    '''Check for equivalence of two sparse dictionaries without safeguards for speed.'''

    @pytest.mark.parametrize('junk_sd',
                             ('junk', 1, [1, 2, 3], {0, 1, 2}, np.pi, None,
                              True, min, np.ndarray)
                             )

    # 1) TEST OUTER SIZE
    if len(DICT1) != len(DICT2): return False

    # 2) TEST INNER SIZES
    for outer_key in DICT1:  # ALREADY ESTABLISHED OUTER KEYS ARE EQUAL
        if len(DICT1[outer_key]) != len(DICT2[outer_key]): return False

    # for outer_key in DICT1:
    #     if not np.array_equiv(unzip_to_ndarray_float64({0: DICT1[outer_key]}),
    #                          unzip_to_ndarray_float64({0: DICT2[outer_key]})): return False

    # 3) TEST INNER KEYS ARE EQUAL
    for outer_key in DICT1:   # ALREADY ESTABLISHED OUTER KEYS ARE EQUAL
        if not np.allclose(np.fromiter(DICT1[outer_key], dtype=np.int32),
                             np.fromiter(DICT2[outer_key], dtype=np.int32), rtol=1e-8, atol=1e-8): return False

    # 4) TEST INNER VALUES ARE EQUAL
    for outer_key in DICT1:  # ALREADY ESTABLISHED OUTER KEYS ARE EQUAL
        if not np.allclose(np.fromiter(DICT1[outer_key].values(), dtype=np.float64),
                          np.fromiter(DICT2[outer_key].values(), dtype=np.float64), rtol=1e-8, atol=1e-8): return False

    # IF GET THIS FAR, MUST BE True
    return True


def safe_sparse_equiv(DICT1, DICT2):
    '''Safely check for equivalence of two sparse dictionaries with safeguards.'''

    @pytest.mark.parametrize('junk_sd',
                             ('junk', 1, [1, 2, 3], {0, 1, 2}, np.pi, None,
                              True, min, np.ndarray)
                             )

    DICT1 = val._dict_init(DICT1)
    DICT2 = val._dict_init(DICT2)
    val._insufficient_dict_args_2(DICT1, DICT2)

    # 1) TEST OUTER KEYS ARE EQUAL
    if not np.allclose(np.fromiter(DICT1, dtype=np.int32),
                      np.fromiter(DICT2, dtype=np.int32), rtol=1e-8, atol=1e-8): return False

    # 2) RUN core_sparse_equiv
    if core_sparse_equiv(DICT1, DICT2) is False: return False

    return True  # IF GET TO THIS POINT, MUST BE TRUE


def return_uniques(DICT1):
    '''Return unique values of a sparse dictionary as list.'''

    @pytest.mark.parametrize('junk_sd',
                             ('junk', 1, [1, 2, 3], {0, 1, 2}, np.pi, None,
                              True, min, np.ndarray)
                             )

    DICT1 = val._dict_init(DICT1)
    val._insufficient_dict_args_1(DICT1)
    NUM_HOLDER, STR_HOLDER = [], []

    for outer_key in DICT1:   # 10/16/22 DONT CHANGE THIS, HAS TO DEAL W DIFF DTYPES, DONT USE np.unique, BLOWS UP FOR '<' not supported between instances of 'str' and 'int'
        for value in DICT1[outer_key].values():
            if True in map(lambda x: x in str(type(value)).upper(), ['INT', 'FLOAT']):
                if value not in NUM_HOLDER: NUM_HOLDER.append(value)
            else:
                if value not in STR_HOLDER: STR_HOLDER.append(str(value))

    if sparsity(DICT1) < 100 and 0 not in NUM_HOLDER: NUM_HOLDER.append(0)

    UNIQUES = np.array(sorted(NUM_HOLDER) + sorted(STR_HOLDER), dtype=object)

    del NUM_HOLDER, STR_HOLDER

    return UNIQUES





