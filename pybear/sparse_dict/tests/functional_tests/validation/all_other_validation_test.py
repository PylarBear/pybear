# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest
import random
from copy import deepcopy
import numpy as np
import pandas as pd
from pybear.sparse_dict import _validation as val
import dask.dataframe as ddf



@pytest.fixture
def good_sd():
    return {
                0: {0:1,1:2},
                1: {0:2,1:0}
    }

@pytest.fixture
def bad_sd():
    return {
                0: {0:1},
                1: {0:0, 1:0}
    }


@pytest.fixture
def good_datadict():
    DATA = np.random.randint(0, 10, (10, 3))
    return {
        'A': DATA[:, 0],
        'B': DATA[:, 1],
        'C': DATA[:, 2],
    }







@pytest.mark.skip(reason="future is uncertain")
def module_name(sys_modules_str):
    """Return module name."""
    return get_module_name(sys_modules_str)



class TestListInit:

    @pytest.mark.parametrize('non_list_data',
         (1, np.pi, True, {'a':1}, min, 'junk', lambda x: x)
    )
    @pytest.mark.parametrize('non_list_header',
         (1, np.pi, True, {'a':1}, min, 'junk', lambda x: x),
    )
    def test_rejects_non_lists(self, non_list_data, non_list_header):
        with pytest.raises(TypeError):
            val._list_init(non_list_data, non_list_header, 'test_module')


    def test_accepts_lists(self):
        data, hdr = val._list_init([[1,2,3]], ['hdr1'])
        assert isinstance(data, np.ndarray)
        assert np.array_equiv(data, [[1,2,3]])
        assert isinstance(hdr, list)
        assert hdr == ['hdr1']


    def test_accepts_none(self):
        data, hdr = val._list_init(None, None)
        assert np.array_equiv(data, np.array([]))
        assert hdr == [[]]

        data, hdr = val._list_init([[1,2,3]], None)
        assert np.array_equiv(data, np.array([[1,2,3]]))
        assert hdr == [[]]


class TestDictInit:

    @pytest.mark.parametrize('non_dict',
                             (1, np.pi, True, [1,2,3], lambda x: x, 'junk'))
    def test_rejects_non_sd(self, non_dict):
        with pytest.raises(TypeError):
            val._dict_init(non_dict)


    def test_accepts_good_sd(self, good_sd):
        assert val._dict_init(good_sd) == good_sd


    def test_rejects_bad_sd(self, bad_sd):
        with pytest.raises(ValueError):
            val._dict_init(bad_sd)


    def test_accepts_none(self):
        assert val._dict_init(None) == {}


class TestDataDictInit:
    @pytest.mark.parametrize('non_df',
                             ('junk', 0, True, np.nan, np.pi, [1, 2, 3],
                              {1, 2, 3}, lambda x: x)
                             )
    def test_rejects_non_datadict(self, non_df):
        with pytest.raises(TypeError):
            val._datadict_init(non_df)


    def test_accepts_datadict(self, good_datadict):
        data, hdr = val._datadict_init(good_datadict)
        assert data == good_datadict
        assert np.array_equiv(hdr, ['A', 'B', 'C'])


    def test_accepts_none(self):
        data, hdr = val._datadict_init(None)
        assert data == {}
        assert hdr == [[]]


    def test_rekeys_datadict(self, good_datadict):
        OUTPUT_DATA, OUTPUT_HDR = val._datadict_init(good_datadict)
        REF_DATADICT = {}
        REF_HDR = [[]]
        wip_datadict = deepcopy(good_datadict)
        for idx, key in enumerate(good_datadict):
            REF_HDR[0].append(key)
            REF_DATADICT[idx] = wip_datadict.pop(key)

        del wip_datadict

        assert np.array_equiv(OUTPUT_HDR, np.array([['A', 'B', 'C']]))
        assert np.array_equiv(list(REF_DATADICT.keys()), [0,1,2])

        for idx, (hdr, values) in enumerate(REF_DATADICT.items()):
            assert np.array_equiv(OUTPUT_DATA[idx], values)


class TestDataFrameInit:

    @pytest.mark.parametrize('non_df',
        ('junk', 0, True, np.nan, np.pi, [1,2,3], {1,2,3}, lambda x: x)
    )
    def test_rejects_non_df(self, non_df):
        with pytest.raises(TypeError):
            val._dataframe_init(non_df)


    def test_accepts_df(self, good_sd):
        df = pd.DataFrame(good_sd)
        data, hdr = val._dataframe_init(df)
        assert df.equals(data)
        assert np.array_equiv(hdr, np.array([[0, 1]]))


    def test_accepts_none(self):
        data, hdr = val._dataframe_init(None)
        assert pd.DataFrame().equals(data)
        assert np.array_equiv(hdr, np.array([[]]))


    def test_rekeys_df(self):
        DATA = np.random.randint(0,10,(10,3))
        COLUMNS = list('ABC')
        DF = pd.DataFrame(data=DATA, columns=COLUMNS)

        OUTPUT_DATA, OUTPUT_HDR = val._dataframe_init(DF)
        assert pd.DataFrame(data=DATA, columns=[0,1,2]).equals(OUTPUT_DATA)
        assert np.array_equiv(OUTPUT_HDR, np.array([['A', 'B', 'C']]))


class TestListCheck:

    @pytest.mark.parametrize('non_list_like',
        (0,True,None,'junk', {'a':1}, np.pi, lambda x: x, min)
    )
    def test_rejects_non_list_like(self, non_list_like):
        with pytest.raises(TypeError):
            val._list_check(non_list_like)

    def test_rejects_empty_list(self):
        with pytest.raises(ValueError):
            val._list_check([])


    def test_rejects_ragged(self):
        with pytest.raises(ValueError):
            val._list_check([[1,2,3], [1,2], [1,2,3]])

    @pytest.mark.parametrize('list_like',
        ([1,2,3], (1,2,3), {1,2,3}, np.random.randint(0,10,(3,3)), np.array([1,2]))
    )
    def test_accepts_list_like(self, list_like):
        val._list_check(list_like)


    def test_reject_more_than_2d(self):
        with pytest.raises(ValueError):
            val._list_check(np.random.randint(0,10,(3,3,3)))


class TestIsSparseOuter:

    @pytest.mark.parametrize('outer_dict',
                                ({0:{0:3,1:4}}, {0:{0:1,1:2},1:{0:2,1:0}})
    )
    def test_returns_true(self, outer_dict):
        assert val._is_sparse_outer(outer_dict)


    @pytest.mark.parametrize('bad_o_d',
        (0, False, None, [1,2,3], (1,2,3), {1,2,3}, np.pi, lambda x: x, min)
    )
    def test_non_dict_returns_false(self, bad_o_d):
        assert not val._is_sparse_outer(bad_o_d)

    @pytest.mark.parametrize('non_o_d',
        ({1:1,2:2,3:3}, {0:1, 3:0}, {})
    )
    def test_returns_false(self, non_o_d):
        assert not val._is_sparse_outer(non_o_d)



class TestIsSparseInner:

    @pytest.mark.parametrize('non_dict',
        (1, False, None, np.pi, [1,2], {1,2}, lambda x: x, min, 'junk')
    )
    def test_non_dict_returns_false(self, non_dict):
        assert not val._is_sparse_inner(non_dict)


    def test_full_sparse_dict_returns_false(self, good_sd):
        assert not val._is_sparse_inner(good_sd)


    def test_inner_dict_returns_true(self):
        assert val._is_sparse_inner({0:1, 1:2, 2:3})
        assert val._is_sparse_inner({0:1, 5:0})


    def test_other_dict_returns_false(self):
        assert not val._is_sparse_inner({'a': 'apple', 'b':'boat'})
        assert not val._is_sparse_inner({0: 'cat', 1: 'hat'})


class TestDataDictCheck:

    @pytest.mark.parametrize('non_dict',
         (0, None, True, np.pi, [1,2,3], {1,2,3}, min, lambda x: x, 'junk')
    )
    def test_rejects_non_dict(self, non_dict):
        with pytest.raises(TypeError):
            val._datadict_check(non_dict)


    def test_rejects_empty_dict(self):
        with pytest.raises(ValueError):
            val._datadict_check({})


    @pytest.mark.parametrize('values',
                                ([1,2,3], {1,2,3}, (1,2,3), np.array([1,2,3]))
    )
    def test_accepts_vector_like(self, values):
        val._datadict_check({'hdr1':values})


    @pytest.mark.parametrize('values',
                                (0, True, None, 'junk', {'a':1}, lambda x: x)
    )
    def test_rejects_non_vector_like(self, values):
        with pytest.raises(TypeError):
            val._datadict_check({'hdr1':values})


    def test_rejects_array(self):
        with pytest.raises(ValueError):
            val._datadict_check({'hdr1':np.random.randint(0,10,(3,3))})



class TestDataFrameCheck:

    def test_accepts_pd_df(self):
        val._dataframe_check(pd.DataFrame())

    def test_accepts_dask_df(self):
        val._dataframe_check(ddf.from_array(np.random.randint(0,10,(5,3))))


    @pytest.mark.parametrize('non_df',
        (0, True, None, np.pi, [1,2,3], (1,2,3), lambda x: x, {'a':1}, 'junk')
    )
    def test_rejects_non_df(self, non_df):
        with pytest.raises(TypeError):
            val._dataframe_check(non_df)



class TestInsufficientListArgs:

    @pytest.mark.parametrize('non_list',
        (0, np.pi, True, None, {'a':1}, min, lambda x: x)
    )
    def test_rejects_non_list_like(self, non_list):
        with pytest.raises(TypeError):
            val._insufficient_list_args_1(non_list)


    @pytest.mark.parametrize('list_type',
        (
        [1,2,3],
        (1,2,3),
        {1,2,3},
        np.random.randint(0,10,(3,3)),
        [[1,2,3],[4,5,6],[7,8,9]]
        )
    )
    def test_accepts_list_like(self, list_type):
        val._insufficient_list_args_1(list_type)


class TestInsufficientDictArgs1:

    @pytest.mark.parametrize('non_dict',
        (0, True, np.pi, np.nan, {1,2,3}, [1,2,3], 'junk', lambda x: x, None)
    )
    def test_rejects_non_dict(self, non_dict):
        with pytest.raises(TypeError):
            val._insufficient_dict_args_1(non_dict)


    def test_rejects_empty_dict(self):
        with pytest.raises(ValueError):
            val._insufficient_dict_args_1({})


    @pytest.mark.parametrize('good_dict',
        ({0:1, 1:2}, {'a':1}, {0:{1:1,2:0}, 1:{0:1, 2:2}})
    )
    def test_accepts_dicts(self, good_dict):
        val._insufficient_dict_args_1(good_dict)


class TestInsufficientDictArgs2:

    @pytest.mark.parametrize('non_dict',
        (0, True, np.pi, np.nan, {1,2,3}, [1,2,3], 'junk', lambda x: x, None)
    )
    def test_rejects_non_dict(self, non_dict, good_sd):
        with pytest.raises(TypeError):
            val._insufficient_dict_args_2(non_dict, good_sd)
            val._insufficient_dict_args_2(good_sd, non_dict)


    def test_rejects_empty_dict(self, good_sd):
        with pytest.raises(ValueError):
            val._insufficient_dict_args_2({}, good_sd)
            val._insufficient_dict_args_2(good_sd, {})
            val._insufficient_dict_args_2({}, {})


    @pytest.mark.parametrize('good_dict1',
        ({0: 1, 1: 2}, {'a': 1}, {0: {1: 1, 2: 0}, 1: {0: 1, 2: 2}})
        )
    @pytest.mark.parametrize('good_dict2',
        ({0: 1, 1: 2}, {'a': 1}, {0: {1: 1, 2: 0}, 1: {0: 1, 2: 2}})
    )
    def test_accepts_dicts(self, good_dict1, good_dict2):
        val._insufficient_dict_args_2(good_dict1, good_dict2)


class TestInsufficientDatadictArgs:
    @pytest.mark.parametrize('non_dict',
        (0, True, np.pi, np.nan, {1,2,3}, [1,2,3], 'junk', lambda x: x, None)
    )
    def test_rejects_non_dict(self, non_dict):
        with pytest.raises(TypeError):
            val._insufficient_datadict_args_1(non_dict)


    def test_rejects_empty_dict(self):
        with pytest.raises(ValueError):
            val._insufficient_datadict_args_1({})


    @pytest.mark.parametrize('bad_dict',
        ({0:1, 1:2}, {'a':1}, {0:{1:1,2:0}, 1:{0:1, 2:2}})
    )
    def test_rejects_bad_dicts(self, bad_dict):
        with pytest.raises(TypeError):
            val._insufficient_datadict_args_1(bad_dict)


    def test_accepts_good_dict(self, good_datadict):
        val._insufficient_datadict_args_1(good_datadict)


class TestInsufficientDataFrameArgs:
    @pytest.mark.parametrize('non_df',
        (0, True, np.pi, np.nan, {1,2,3}, [1,2,3], 'junk', lambda x: x, None)
    )
    def test_rejects_non_df(self, non_df):
        with pytest.raises(TypeError):
            val._insufficient_dataframe_args_1(non_df)


    def test_rejects_empty_df(self):
        with pytest.raises(ValueError):
            val._insufficient_dataframe_args_1(pd.DataFrame({}))


    def test_accepts_good_pd_df(self):
        df = pd.DataFrame(np.random.randint(0,10,(5,3)), columns=list('abc'))
        val._insufficient_dataframe_args_1(df)


    def test_accepts_good_dask_df(self):
        df = ddf.from_array(np.random.randint(0,10,(5,3)), columns=list('abc'))
        val._insufficient_dataframe_args_1(df)





###############################################################################
# LINALG CHECKS & EXCEPTIONS ##################################################

class TestDotSizeCheck:

    @pytest.mark.parametrize('non_dict',
         (None, np.pi, [1], {1}, min, 'junk', lambda x: x, True, 1)
    )
    def test_rejects_non_dict(self, non_dict):
        with pytest.raises(TypeError):
            val._dot_size_check(non_dict)

    def test_accepts_vector_same_len(self):
        dict1 = {0: {0:1, 1:2, 2:0}}
        dict2 = {1: {1:2, 2:2}}
        val._dot_size_check(dict1, dict2)


    def test_rejects_array(self):
        dict1 = {0: {0:1, 1:2, 2:0}, 1:{1:1,2:1}}
        dict2 = {1: {1:2, 2:2}, 2:{0:2,2:0}}
        with pytest.raises(ValueError):
            val._dot_size_check(dict1, dict2)


    def rejects_diff_len(self):
        dict1 = {0: {0:1, 1:2}}
        dict2 = {0: {1:2, 2:2}}
        with pytest.raises(ValueError):
            val._dot_size_check(dict1, dict2)


class TestBroadcastCheck:

    @pytest.mark.parametrize('non_dict',
         (None, np.pi, [1], {1}, min, 'junk', lambda x: x, True, 1)
    )
    def test_rejects_non_dict(self, non_dict):
        with pytest.raises(TypeError):
            val._broadcast_check(non_dict)

    @pytest.mark.parametrize('i', (1, 3))
    @pytest.mark.parametrize('j', (1, 3))
    @pytest.mark.parametrize('k', (2, 4))
    def test_accepts_good(self, i, j, k):
        dict1 = {_: {__: random.randint(1, 4) for __ in range(j)} for _ in range(i)}
        dict2 = {_: {__: random.randint(1, 4) for __ in range(k)} for _ in range(j)}
        val._broadcast_check(dict1, dict2)

    @pytest.mark.parametrize('i', (1, 3))
    @pytest.mark.parametrize('j', (1, 3))
    @pytest.mark.parametrize('k', (2, 4))
    def test_rejects_bad(self, i, j, k):
        dict1 = {_: {__: random.randint(1, 4) for __ in range(i)} for _ in range(j)}
        dict2 = {_: {__: random.randint(1, 4) for __ in range(j)} for _ in range(k)}
        with pytest.raises(ValueError):
            val._broadcast_check(dict1, dict2)


class TestMatrixShapeCheck:

    @pytest.mark.parametrize('non_dict',
         (None, np.pi, [1], {1}, min, 'junk', lambda x: x, True, 1)
    )
    def test_rejects_non_dict(self, non_dict):
        with pytest.raises(TypeError):
            val._matrix_shape_check(non_dict)

    @pytest.mark.parametrize('i', (1, 2, 3))
    @pytest.mark.parametrize('j', (1, 2, 3))
    def test_accepts_good(self, i, j):
        dict1 = {_: {__: random.randint(1, 4) for __ in range(i)} for _ in range(j)}
        dict2 = {_: {__: random.randint(1, 4) for __ in range(i)} for _ in range(j)}
        val._matrix_shape_check(dict1, dict2)

    @pytest.mark.parametrize('i', (1, 3, 5))
    @pytest.mark.parametrize('j', (2, 4, 6))
    def test_rejects_bad(self, i, j):
        dict1 = {_: {__: random.randint(1, 4) for __ in range(i)} for _ in range(i)}
        dict2 = {_: {__: random.randint(1, 4) for __ in range(j)} for _ in range(j)}
        with pytest.raises(ValueError):
            val._matrix_shape_check(dict1, dict2)


class TestOuterLenCheck:

    @pytest.mark.parametrize('non_dict',
         (None, np.pi, [1], {1}, min, 'junk', lambda x: x, True, 1)
    )
    def test_rejects_non_dict(self, non_dict):
        with pytest.raises(TypeError):
            val._outer_len_check(non_dict)

    @pytest.mark.parametrize('i', (1, 2, 3))
    def test_accepts_good(self, i):
        dict1 = {_: {__: random.randint(1, 4) for __ in range(3)} for _ in range(i)}
        dict2 = {_: {__: random.randint(1, 4) for __ in range(4)} for _ in range(i)}
        val._outer_len_check(dict1, dict2)

    @pytest.mark.parametrize('i', (1, 3, 5))
    @pytest.mark.parametrize('j', (2, 4, 6))
    def test_rejects_bad(self, i, j):
        dict1 = {_: {__: random.randint(1, 4) for __ in range(3)} for _ in range(i)}
        dict2 = {_: {__: random.randint(1, 4) for __ in range(4)} for _ in range(j)}
        with pytest.raises(ValueError):
            val._outer_len_check(dict1, dict2)


class TestInnerLenCheck:
    @pytest.mark.parametrize('non_dict',
        (None, np.pi, [1], {1}, min, 'junk', lambda x: x, True, 1)
    )
    def test_rejects_non_dict(self, non_dict):
        with pytest.raises(TypeError):
            val._inner_len_check(non_dict)


    @pytest.mark.parametrize('i', (1, 2, 3))
    def test_accepts_good(self, i):
        dict1 = {_: {__: random.randint(1, 4) for __ in range(i)} for _ in range(3)}
        dict2 = {_: {__: random.randint(1, 4) for __ in range(i)} for _ in range(4)}
        val._inner_len_check(dict1, dict2)


    @pytest.mark.parametrize('i', (1, 3, 5))
    @pytest.mark.parametrize('j', (2, 4, 6))
    def test_rejects_bad(self, i, j):
        dict1 = {_: {__: random.randint(1, 4) for __ in range(i)} for _ in range(3)}
        dict2 = {_: {__: random.randint(1, 4) for __ in range(j)} for _ in range(4)}
        with pytest.raises(ValueError):
            val._inner_len_check(dict1, dict2)



class TestSymmetricMatmulCheck:

    @pytest.fixture
    def valid_dict1(self):
        return 1

    @pytest.fixture
    def valid_dict1_t(self):
        return 1

    @pytest.fixture
    def valid_dict1_t(self):
        return 1

    @pytest.fixture
    def valid_dict2(self):
        return 1

    @pytest.fixture
    def valid_dict1(self):
        return 1


    @pytest.mark.parametrize('non_dict',
         (None, np.pi, [1], {1}, min, 'junk', lambda x: x, True, 1)
    )
    def test_rejects_non_dict(self, non_dict):
        with pytest.raises(TypeError):
            val._inner_len_check(non_dict)


    def test_correctly_returns_true(self):
        dict1 = {0:{0:1, 1:0}, 1:{1:1}}
        assert val._symmetric_matmul_check(dict1, dict1)


    def test_correctly_returns_false(self):
        dict1 = {0:{0:1, 1:2}, 1:{0:3, 1:1}}
        dict2 = {0:{0:3, 1:1}, 1:{0:2, 1:0}}
        with pytest.raises(ValueError):
            assert not val._symmetric_matmul_check(dict1, dict2)


    def test_accepts_valid_dict1_transpose(self):


    def test_rejects_bad_dict1_transpose(self):


    def test_accepts_valid_dict2_transpose(self):


    def test_rejects_bad_dict2_transpose(self):


# END LINALG CHECKS & EXCEPTIONS #############################################
###############################################################################

"""






