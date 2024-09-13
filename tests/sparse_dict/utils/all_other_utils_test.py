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
                                        safe_sparse_equiv,
                                        return_uniques,
                                        drop_placeholders,
                                        dtype_,
                                        astype
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
def good_inner_dict():
    return {0:1, 1:2, 2:3}






class TestOuterLen:
    @pytest.mark.parametrize('junk_sd',
        ('junk', 1, [1, 2, 3], {0, 1, 2}, np.pi, None, True, min, np.ndarray)
    )
    def test_rejects_non_sd(self, junk_sd):
        with pytest.raises(TypeError):
            outer_len(junk_sd)


    def test_rejects_bad_sd(self, bad_sd_1):
        with pytest.raises(ValueError):
            outer_len(bad_sd_1)


    def test_rejects_inner_dict(self, good_inner_dict):
        with pytest.raises(ValueError):
            outer_len(good_inner_dict)


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

    @pytest.mark.xfail(reason='does not have _validation to handle bad dict')
    def test_rejects_bad_sd1(self, bad_sd_1):
        with pytest.raises(ValueError):
            inner_len_quick(bad_sd_1)


    def test_accepts_inner_dict(self, good_inner_dict):
        inner_len_quick(good_inner_dict)


    def test_accuracy(self, good_sd_1, good_sd_2, good_inner_dict):
        assert inner_len_quick(good_sd_1) == 3
        assert inner_len_quick(good_sd_2) == 4
        assert inner_len_quick(good_inner_dict) == 3


class TestInnerLen:
    @pytest.mark.parametrize('junk_sd',
        ('junk', 1, [1, 2, 3], {0, 1, 2}, np.pi, None, True, min, np.ndarray)
    )
    def test_rejects_non_sd(self, junk_sd):
        with pytest.raises(TypeError):
            inner_len(junk_sd)


    def test_rejects_bad_sd(self, bad_sd_1):
        with pytest.raises(ValueError):
            inner_len(bad_sd_1)


    def test_accepts_inner_dict(self, good_inner_dict):
        inner_len(good_inner_dict)


    def test_accuracy(self, good_sd_1, good_sd_2, good_inner_dict):
        assert inner_len(good_sd_1) == 3
        assert inner_len(good_sd_2) == 4
        assert inner_len(good_inner_dict) == 3


class TestSize:

    @pytest.mark.parametrize('junk_sd',
        ('junk', 1, [1, 2, 3], {0, 1, 2}, np.pi, None, True, min, np.ndarray)
    )
    def test_rejects_non_sd(self, junk_sd):
        with pytest.raises(TypeError):
            size_(junk_sd)


    def test_rejects_bad_sd(self, bad_sd_1):
        with pytest.raises(ValueError):
            size_(bad_sd_1)


    def test_accepts_inner_dict(self, good_inner_dict):
        size_(good_inner_dict)


    def test_accuracy(self, good_sd_1, good_sd_2, good_inner_dict):
        assert size_(good_sd_1) == 12
        assert size_(good_sd_2) == 4
        assert size_(good_inner_dict) == 3


class TestShape:

    @pytest.mark.parametrize('junk_sd',
        ('junk', 1, [1, 2, 3], {0, 1, 2}, np.pi, None, True, min, np.ndarray)
    )
    def test_rejects_non_sd(self, junk_sd):
        if junk_sd is None:
            pytest.xfail(reason=f'has numpy-like handling for None')

        with pytest.raises(TypeError):
            shape_(junk_sd)


    def test_rejects_bad_sd(self, bad_sd_1):
        with pytest.raises(ValueError):
            shape_(bad_sd_1)


    def test_accepts_inner_dict(self, good_inner_dict):
        assert shape_(good_inner_dict) == (3, )


    def test_accuracy(self, good_sd_1, good_sd_2):
        assert shape_(good_sd_1) == (4, 3)
        assert shape_(good_sd_2) == (1, 4)


    def test_accuracy2(self):
        assert shape_({0:{}, 1:{}}) == (2, 0)


    def test_accuracy3(self):
        assert shape_({0: {}}) == (1, 0)


    def test_accuracy4(self):
        assert shape_({0: 1, 1: 2}) == (2, )


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


    def test_cleans_a_bad_inner_dict(self):
        assert clean({0:0,1:0,2:1}) == {2:1}


class TestSparsity:

    @pytest.mark.parametrize('junk_sd',
        ('junk', 1, [1, 2, 3], {0, 1, 2}, np.pi, None, True, min, np.ndarray)
    )
    def test_rejects_non_sd(self, junk_sd):
            with pytest.raises(TypeError):
                sparsity(junk_sd)


    def test_rejects_bad_sd(self, bad_sd_1):
        with pytest.raises(ValueError):
            sparsity(bad_sd_1)

    def test_accepts_inner_dict(self, good_inner_dict):
            sparsity(good_inner_dict)


    def test_accuracy(self, good_sd_1, good_sd_2, good_inner_dict):
        assert round(sparsity(good_sd_1), 1) == 41.7
        assert sparsity(good_sd_2) == 0
        assert sparsity({0:{10:0}}) == 100
        assert sparsity(good_inner_dict) == 0
        assert sparsity({0:1, 2:1, 3:0}) == 50


class TestCoreSparseEquiv:

    # ONLY DO VALIDATION ON safe_sparse_equiv

    def test_accepts_inner_dict(self, good_sd_1, good_inner_dict):
        core_sparse_equiv(good_inner_dict, good_sd_1)
        core_sparse_equiv(good_sd_1, good_inner_dict)
        core_sparse_equiv(good_inner_dict, good_inner_dict)


    def test_accuracy(self, good_sd_1, good_sd_2, good_inner_dict):

        assert core_sparse_equiv(good_sd_1, good_sd_1)
        assert core_sparse_equiv(good_sd_2, good_sd_2)
        assert core_sparse_equiv(good_inner_dict, good_inner_dict)

        assert not core_sparse_equiv(good_sd_1, good_sd_2)
        assert not core_sparse_equiv(good_sd_1, good_inner_dict)
        assert not core_sparse_equiv(good_sd_2, good_inner_dict)


class TestSafeSparseEquiv:

    @pytest.mark.parametrize('non_sd',
        (0, np.pi, False, min, lambda x: x, {1,2}, [1,2], (1,2))
    )
    def test_rejects_non_sd(self, non_sd, good_sd_1):
        with pytest.raises(TypeError):
            safe_sparse_equiv(non_sd, good_sd_1)

        with pytest.raises(TypeError):
            safe_sparse_equiv(good_sd_1, non_sd)


    @pytest.mark.parametrize('junk_sd',
                 ({'a':1}, {np.pi: 'a'}, {np.nan: None})
    )
    def test_rejects_junk_sd(self, junk_sd, good_sd_1):
        with pytest.raises(ValueError):
            safe_sparse_equiv(junk_sd, good_sd_1)

        with pytest.raises(ValueError):
            safe_sparse_equiv(good_sd_1, junk_sd)


    def test_rejects_bad_sd(self, good_sd_1, bad_sd_1):
        with pytest.raises(ValueError):
            safe_sparse_equiv(good_sd_1, bad_sd_1)

        with pytest.raises(ValueError):
            safe_sparse_equiv(bad_sd_1, good_sd_1)


    def test_accepts_inner_dict(self, good_inner_dict, good_sd_1):
        safe_sparse_equiv(good_inner_dict, good_sd_1)
        safe_sparse_equiv(good_sd_1, good_inner_dict)
        safe_sparse_equiv(good_inner_dict, good_inner_dict)


    def test_accuracy(self, good_sd_1, good_sd_2, good_inner_dict):

        assert safe_sparse_equiv(good_sd_1, good_sd_1)
        assert safe_sparse_equiv(good_sd_2, good_sd_2)
        assert safe_sparse_equiv(good_inner_dict, good_inner_dict)

        assert not safe_sparse_equiv(good_sd_1, good_sd_2)
        assert not safe_sparse_equiv(good_sd_1, good_inner_dict)
        assert not safe_sparse_equiv(good_sd_2, good_inner_dict)




class TestReturnUniques:

    @pytest.mark.parametrize('junk_sd',
        ('junk', 1, [1, 2, 3], {0, 1, 2}, np.pi, None, True, min, np.ndarray)
    )
    def test_rejects_non_sd(self, junk_sd):
        with pytest.raises(TypeError):
            return_uniques(junk_sd)


    def test_rejects_bad_sd(self, bad_sd_1):
        with pytest.raises(ValueError):
            return_uniques(bad_sd_1)


    def test_rejects_bad_sd_2(self):
        with pytest.raises(TypeError):
            return_uniques({0: {0:'a', 1:'b', 2:'c', 3:'d'}})


    def test_accepts_inner_dict(self, good_inner_dict):
        return_uniques(good_inner_dict)


    def test_accuracy(self, good_sd_1, good_sd_2, good_inner_dict):
        assert np.array_equiv(return_uniques(good_sd_1), np.array([0, 1, 2, 3]))
        assert np.array_equiv(return_uniques(good_sd_2), np.array([1, 2, 3, 4]))
        assert np.array_equiv(return_uniques(good_inner_dict), np.array([1, 2, 3]))




class TestDropPlaceholders:

    @pytest.mark.parametrize('non_dict',
        (0, np.pi, False, min, lambda x: x, {1, 2}, [1, 2], (1, 2))
    )
    def test_rejects_non_dict(self, non_dict):
        with pytest.raises(TypeError):
            drop_placeholders(non_dict)


    def test_rejects_bad_sd(self, bad_sd_1):
        with pytest.raises(ValueError):
            drop_placeholders(bad_sd_1)


    def test_accepts_outer_dict(self):
        assert drop_placeholders({0:{0:1,1:2,2:3}}) == {0:{0:1,1:2,2:3}}

    def test_accepts_inner_dict(self):
        assert drop_placeholders({0:1,1:2,2:3}) == {0:1,1:2,2:3}

    def test_accuracy(self):

        assert drop_placeholders({0:{0:1,1:0},1:{1:0}}) == {0:{0:1}, 1:{}}

        assert drop_placeholders({0:1,1:2,2:0}) == {0:1,1:2}


class TestDtype_:

    @pytest.mark.parametrize('non_dict',
        (0, np.pi, False, min, lambda x: x, {1, 2}, [1, 2], (1, 2))
    )
    def test_rejects_non_dict(self, non_dict):
        with pytest.raises(TypeError):
            dtype_(non_dict)


    def test_rejects_bad_sd(self, bad_sd_1):
        with pytest.raises(ValueError):
            dtype_(bad_sd_1)


    def test_accepts_outer_dict(self):
        dtype_({0:{0:1,1:2,2:3}})


    def test_accepts_inner_dict(self):
        dtype_({0:1,1:2,2:3})


    def test_rejects_multiple_dtypes(self):
        with pytest.raises(ValueError):
            dtype_({0:1, 1:np.pi})

        with pytest.raises(ValueError):
            dtype_({0:{0:0, 1:np.pi}})


    def rejects_empty_sparse_dict(self):

        with pytest.raises(ValueError):
            dtype_({0: {}})

        with pytest.raises(ValueError):
            dtype_({})


    def test_accuracy(self):

        assert dtype_({0:1,1:2,2:3}) is int

        assert dtype_({0:{i:np.float64(np.pi) for i in range(5)}}) is np.float64

        assert dtype_({i:np.float64(np.pi) for i in range(5)}) is np.float64

        assert dtype_({0:{0:np.float64(np.pi)}}) is np.float64

        assert dtype_({0: {0:3.14,1:2.718,2:2.0000}}) is float

        assert dtype_({0:{0:0}, 1:{0:0}, 2:{0:0}, 3:{0:0}})


class TestAstype:

    @pytest.mark.parametrize('non_dict',
        (0, np.pi, False, min, lambda x: x, {1, 2}, [1, 2], (1, 2))
    )
    def test_rejects_non_dict(self, non_dict):
        with pytest.raises(TypeError):
            astype(non_dict, dtype=np.float64)


    def test_rejects_bad_sd(self, bad_sd_1):
        with pytest.raises(ValueError):
            astype(bad_sd_1)


    def test_accepts_outer_dict(self):
        astype({0:{0:1,1:2,2:3}})


    def test_accepts_inner_dict(self):
        astype({0:1,1:2,2:3})


    def rejects_empty_sparse_dict(self):

        with pytest.raises(ValueError):
            dtype_({0: {}})

        with pytest.raises(ValueError):
            dtype_({})


    def test_accuracy(self):

        assert dtype_(astype({0: 1, 1: 2, 2: 3}, dtype=np.float64)) is np.float64

        assert dtype_(astype({0: 1, 1: 2, 2: 3}, dtype=int)) is int

        assert dtype_(astype({0: 1, 1: 2, 2: 3}, dtype=float)) is float

        assert dtype_(astype({0: {0: 1, 1: 2, 2: 3}}, dtype=np.float64)) is np.float64

        assert dtype_(astype({0: {0: 1, 1: 2, 2: 3}}, dtype=int)) is int

        assert dtype_(astype({0: {0: 1, 1: 2, 2: 3}}, dtype=float)) is float








