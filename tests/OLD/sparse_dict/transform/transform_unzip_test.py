# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest
from sparse_dict._transform import (

                                    unzip_to_ndarray,
                                    unzip_to_list,
                                    unzip_to_dask_array,
                                    unzip_to_dataframe,
                                    unzip_to_dask_dataframe,
                                    unzip_to_dense_dict,
                                    unzip_to_datadict
)

from sparse_dict._utils import shape_, outer_len, inner_len

import numpy as np
import pandas as pd
from dask import dataframe as ddf
from dask import array as da



@pytest.fixture
def good_sd_1():
    return {
            0:{0:1,1:2,2:0},
            1:{1:1,2:1}
    }


@pytest.fixture
def unzipped_good_sd_1():
    return np.array(
                        [
                                [1,2,0],
                                [0,1,1]
                        ]
            )


@pytest.fixture
def good_inner_dict():
    return {1:2, 2:0}


@pytest.fixture
def unzipped_good_inner_dict():
    return np.array([0,2,0])





class TestUnzipNdarray:

    @pytest.mark.parametrize('non_sd',
        (0, 1, np.nan, np.pi, [1,2], {1,2}, (1,2), True, False, None,
         min, lambda x: x, np.float64, 'junk', {np.nan, np.nan})
    )
    def test_rejects_non_sparse_dicts(self, non_sd):
        with pytest.raises(TypeError):
            unzip_to_ndarray(non_sd)


    @pytest.mark.parametrize('bad_sd', ({'a':1}, {0: 'a'}))
    def test_rejects_bad_sparse_dicts(self, bad_sd):
        with pytest.raises(ValueError):
            unzip_to_ndarray(bad_sd)


    def test_accepts_full_sparse_dict(self, good_sd_1):
        unzip_to_ndarray(good_sd_1)


    def test_accepts_inner_sparse_dict(self, good_inner_dict):
        unzip_to_ndarray(good_inner_dict)


    @pytest.mark.parametrize('bad_dtype',
        (0, True, np.nan, np.pi, lambda x: x, {'a': 1}, 'junk', min)
    )
    def test_rejects_bad_dtype(self, bad_dtype, good_sd_1):
        with pytest.raises(TypeError):
            unzip_to_ndarray(good_sd_1, dtype=bad_dtype)


    @pytest.mark.parametrize('good_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    def test_accepts_good_dtype(self, good_dtype, good_sd_1):
        unzip_to_ndarray(good_sd_1, dtype=good_dtype)


    @pytest.mark.parametrize('good_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    def test_accuracy(self, good_dtype, good_sd_1, good_inner_dict,
                      unzipped_good_sd_1, unzipped_good_inner_dict):
        NDARRAY1 = unzip_to_ndarray(good_sd_1, good_dtype)
        assert NDARRAY1.shape == shape_(good_sd_1)
        assert np.array_equiv(NDARRAY1, unzipped_good_sd_1.astype(good_dtype))
        assert NDARRAY1.dtype == good_dtype

        NDARRAY2 = unzip_to_ndarray(good_inner_dict, good_dtype)
        assert NDARRAY2.shape == shape_(good_inner_dict)
        assert np.array_equiv(NDARRAY2, unzipped_good_inner_dict.astype(good_dtype))
        assert NDARRAY2.dtype == good_dtype


    def test_accuracy_empty(self):
        assert np.array_equiv(unzip_to_ndarray({}, dtype=np.uint8),
                              np.array([], dtype=np.uint8))
        assert np.array_equiv(unzip_to_ndarray({0:{}}, dtype=np.uint8),
                              np.array([[]], dtype=np.uint8))
        assert unzip_to_ndarray({0:{}}, dtype=np.uint8).dtype == np.uint8


class TestUnzipList:

    @pytest.mark.parametrize('non_sd',
        (0, 1, np.nan, np.pi, [1,2], {1,2}, (1,2), True, False, None,
         min, lambda x: x, np.float64, 'junk', {np.nan, np.nan})
    )
    def test_rejects_non_sparse_dicts(self, non_sd):
        with pytest.raises(TypeError):
            unzip_to_list(non_sd)


    @pytest.mark.parametrize('bad_sd', ({'a':1}, {0: 'a'}))
    def test_rejects_bad_sparse_dicts(self, bad_sd):
        with pytest.raises(ValueError):
            unzip_to_list(bad_sd)


    def test_accepts_full_sparse_dict(self, good_sd_1):
        unzip_to_list(good_sd_1)


    def test_accepts_inner_sparse_dict(self, good_inner_dict):
        unzip_to_list(good_inner_dict)


    @pytest.mark.parametrize('bad_dtype',
        (0, True, np.nan, np.pi, lambda x: x, {'a': 1}, 'junk', min)
    )
    def test_rejects_bad_dtype(self, bad_dtype, good_sd_1):
        with pytest.raises(TypeError):
            unzip_to_list(good_sd_1, dtype=bad_dtype)


    @pytest.mark.parametrize('good_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    def test_accepts_good_dtype(self, good_dtype, good_sd_1):
        unzip_to_list(good_sd_1, dtype=good_dtype)


    @pytest.mark.parametrize('good_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    def test_accuracy(self, good_dtype, good_sd_1, good_inner_dict,
                      unzipped_good_sd_1, unzipped_good_inner_dict):
        NDARRAY1 = unzip_to_list(good_sd_1, good_dtype)
        assert len(NDARRAY1) == outer_len(good_sd_1)
        assert len(NDARRAY1[0]) == inner_len(good_sd_1)
        assert np.array_equiv(NDARRAY1, unzipped_good_sd_1.astype(good_dtype))
        assert str(good_dtype).upper() in str(type(NDARRAY1[0][0])).upper()

        NDARRAY2 = unzip_to_list(good_inner_dict, good_dtype)
        assert len(NDARRAY2) == inner_len(good_inner_dict)
        assert np.array_equiv(NDARRAY2, unzipped_good_inner_dict.astype(good_dtype))
        assert str(good_dtype).upper() in str(type(NDARRAY2[0])).upper()


    def test_accuracy_empty(self):
        assert unzip_to_list({}) == []
        assert unzip_to_list({0:{}}) == [[]]


class TestUnzipDaskArray:

    @pytest.mark.parametrize('non_sd',
        (0, 1, np.nan, np.pi, [1,2], {1,2}, (1,2), True, False, None,
         min, lambda x: x, np.float64, 'junk', {np.nan, np.nan})
    )
    def test_rejects_non_sparse_dicts(self, non_sd):
        with pytest.raises(TypeError):
            unzip_to_dask_array(non_sd)


    @pytest.mark.parametrize('bad_sd', ({'a':1}, {0: 'a'}))
    def test_rejects_bad_sparse_dicts(self, bad_sd):
        with pytest.raises(ValueError):
            unzip_to_dask_array(bad_sd)


    def test_accepts_full_sparse_dict(self, good_sd_1):
        unzip_to_dask_array(good_sd_1)


    def test_accepts_inner_sparse_dict(self, good_inner_dict):
        unzip_to_dask_array(good_inner_dict)


    @pytest.mark.parametrize('bad_dtype',
        (0, True, np.nan, np.pi, lambda x: x, {'a': 1}, 'junk', min)
    )
    def test_rejects_bad_dtype(self, bad_dtype, good_sd_1):
        with pytest.raises(TypeError):
            unzip_to_dask_array(good_sd_1, dtype=bad_dtype)


    @pytest.mark.parametrize('good_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    def test_accepts_good_dtype(self, good_dtype, good_sd_1):
        unzip_to_dask_array(good_sd_1, dtype=good_dtype)


    @pytest.mark.parametrize('junk_chunk',
        ([1,2], {1,2}, [1,], 1, 'junk', {'a':1}, lambda x: x, 0, True, np.pi)
    )
    def test_rejects_junk_chunks(self, good_sd_1, junk_chunk):
        with pytest.raises(TypeError):
            unzip_to_dask_array(good_sd_1, chunks=junk_chunk)

    @pytest.mark.parametrize('bad_chunk',
                             ((), (1,2,3), (1,2,3,4), (0, ), (0, 0))
    )
    def test_rejects_bad_chunks(self, good_sd_1, bad_chunk):
        with pytest.raises(ValueError):
            unzip_to_dask_array(good_sd_1, chunks=bad_chunk)


    @pytest.mark.parametrize('good_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    def test_accuracy(self, good_dtype, good_sd_1, good_inner_dict,
                      unzipped_good_sd_1, unzipped_good_inner_dict):
        DASK_ARRAY1 = unzip_to_dask_array(good_sd_1, good_dtype, chunks=(1,3))
        assert DASK_ARRAY1.shape == shape_(good_sd_1)
        assert np.array_equiv(DASK_ARRAY1.compute(), unzipped_good_sd_1)
        assert DASK_ARRAY1.dtype == good_dtype
        assert DASK_ARRAY1.chunks == ((1, 1), (3,))

        DASK_ARRAY2 = unzip_to_dask_array(good_inner_dict, good_dtype, chunks=(2,))
        assert DASK_ARRAY2.shape == shape_(good_inner_dict)
        assert np.array_equiv(DASK_ARRAY2.compute(), unzipped_good_inner_dict)
        assert DASK_ARRAY2.dtype == good_dtype
        assert DASK_ARRAY2.chunks == ((2,1),)


    def test_accuracy_empty(self):
        assert np.array_equiv(unzip_to_dask_array({}, dtype=np.float64).compute(),
                              da.array([], dtype=np.float64).compute())
        assert np.array_equiv(unzip_to_dask_array({0:{}}, dtype=np.float64).compute(),
                        da.array([[]], dtype=np.float64).compute())
        assert unzip_to_dask_array({0: {}}, dtype=np.float64).dtype == np.float64


class TestUnzipDatadict:

    @pytest.mark.parametrize('non_sd',
        (0, 1, np.nan, np.pi, [1,2], {1,2}, (1,2), True, False, None,
         min, lambda x: x, np.float64, 'junk', {np.nan, np.nan})
    )
    def test_rejects_non_sparse_dicts(self, non_sd):
        with pytest.raises(TypeError):
            unzip_to_datadict(non_sd)


    @pytest.mark.parametrize('bad_sd', ({'a':1}, {0: 'a'}))
    def test_rejects_bad_sparse_dicts(self, bad_sd):
        with pytest.raises(ValueError):
            unzip_to_datadict(bad_sd)


    def test_accepts_full_sparse_dict(self, good_sd_1):
        unzip_to_datadict(good_sd_1)


    def test_accepts_inner_sparse_dict(self, good_inner_dict):
        unzip_to_datadict(good_inner_dict)


    @pytest.mark.parametrize('bad_header',
        (0, True, np.nan, np.pi, lambda x: x, {'a': 1}, 'junk', min, np.float64)
    )
    def test_rejects_bad_header_type(self, bad_header, good_sd_1):
        with pytest.raises(TypeError):
            unzip_to_datadict(good_sd_1, HEADER=bad_header, dtype=float)


    @pytest.mark.parametrize('bad_header', (list(), list('A'), list('ABCDEFGHI')))
    def test_rejects_bad_header_len(self, bad_header, good_sd_1):
        with pytest.raises(ValueError):
            unzip_to_datadict(good_sd_1, HEADER=bad_header, dtype=float)


    @pytest.mark.parametrize('good_header',
        (['a', 'b', 'c'], [['a', 'b', 'c']], np.array(['a', 'b', 'c']),
         {'a', 'b', 'c'}, ('a', 'b', 'c'), ['a', 'b'], [['a', 'b']])
    )
    def accepts_good_header(self, good_header, good_sd_1):
        unzip_to_datadict(good_sd_1, HEADER=good_header, dtype=float)


    @pytest.mark.parametrize('bad_dtype',
        (0, True, np.nan, np.pi, lambda x: x, {'a': 1}, 'junk', min)
    )
    def test_rejects_bad_dtype(self, bad_dtype, good_sd_1):
        with pytest.raises(TypeError):
            unzip_to_datadict(good_sd_1, dtype=bad_dtype)


    @pytest.mark.parametrize('good_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    def test_accepts_good_dtype(self, good_dtype, good_sd_1):
        unzip_to_datadict(good_sd_1, dtype=good_dtype)


    @pytest.mark.parametrize('good_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    def test_accuracy_no_header(self, good_dtype, good_sd_1, good_inner_dict,
                      unzipped_good_sd_1, unzipped_good_inner_dict):

        REF_DATA1 = unzipped_good_sd_1
        sd1_cols = REF_DATA1.shape[1]

        REF_DATA2 = unzipped_good_inner_dict.reshape((1,-1))
        sd2_cols = 1

        DATADICT1 = unzip_to_datadict(good_sd_1, dtype=good_dtype)

        assert len(DATADICT1) == sd1_cols
        assert np.array_equiv(list(DATADICT1.keys()),
                              list(map(str, range(sd1_cols)))
                              )
        for idx, key in enumerate(DATADICT1):
            assert np.array_equiv(DATADICT1[key], REF_DATA1[:, idx])
            assert DATADICT1[key][0].dtype == good_dtype

        DATADICT2 = unzip_to_datadict(good_inner_dict, dtype=good_dtype)
        assert len(DATADICT2) == sd2_cols
        assert np.array_equiv(list(DATADICT2.keys()),
                              list(map(str, range(sd2_cols)))
                              )
        for idx, key in enumerate(DATADICT2):
            assert np.array_equiv(DATADICT2[key], REF_DATA2[idx])
            assert DATADICT2[key][0].dtype == good_dtype


    @pytest.mark.parametrize('good_header',
        (['a', 'b', 'c'], [['a', 'b', 'c']], np.array(['a', 'b', 'c']),
         {'a', 'b', 'c'}, ('a', 'b', 'c'))
    )
    @pytest.mark.parametrize('good_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    def test_accuracy_header1(self, good_header, good_dtype, good_sd_1,
              good_inner_dict, unzipped_good_sd_1, unzipped_good_inner_dict):

        REF_DATA1 = unzipped_good_sd_1
        sd1_cols = REF_DATA1.shape[1]

        REF_DATA2 = unzipped_good_inner_dict.reshape((1,-1))
        sd2_cols = REF_DATA2.shape[1]

        DATADICT1 = unzip_to_datadict(good_sd_1, good_header, good_dtype)
        assert len(DATADICT1) == sd1_cols
        assert np.array_equiv(list(DATADICT1.keys()), list(good_header))

        for idx, key in enumerate(DATADICT1):
            assert np.array_equiv(DATADICT1[key], REF_DATA1[:, idx])
            assert DATADICT1[key][0].dtype == good_dtype

        DATADICT2 = unzip_to_datadict(good_inner_dict, good_header, good_dtype)
        assert len(DATADICT2) == sd2_cols
        assert np.array_equiv(list(DATADICT2.keys()), list(good_header))
        for idx, key in enumerate(DATADICT2):
            assert np.array_equiv(DATADICT2[key], REF_DATA2[:, idx])
            assert DATADICT2[key][0].dtype == good_dtype


    @pytest.mark.parametrize('good_header',
        (['a', 'b'], [['a', 'b']], np.array(['a', 'b']), {'a', 'b'}, ('a', 'b'))
    )
    @pytest.mark.parametrize('good_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    def test_accuracy_sd1_header2(self, good_header, good_dtype, good_sd_1,
                                unzipped_good_sd_1):

        REF_DATA1 = unzipped_good_sd_1
        sd1_cols = REF_DATA1.shape[0]

        DATADICT1 = unzip_to_datadict(good_sd_1, good_header, good_dtype)
        assert len(DATADICT1) == sd1_cols
        assert np.array_equiv(list(DATADICT1.keys()), list(good_header))

        for idx, key in enumerate(DATADICT1):
            assert np.array_equiv(DATADICT1[key], REF_DATA1[idx])
            assert DATADICT1[key][0].dtype == good_dtype


    @pytest.mark.parametrize('good_header',
        (['a'], [['a']], np.array(['a']), {'a'}, ('a',))
    )
    @pytest.mark.parametrize('good_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    def test_accuracy_sd2_header2(self, good_header, good_dtype, good_sd_1,
              good_inner_dict, unzipped_good_sd_1, unzipped_good_inner_dict):

        REF_DATA2 = unzipped_good_inner_dict.reshape((1,-1))
        sd2_cols = REF_DATA2.shape[0]

        DATADICT2 = unzip_to_datadict(good_inner_dict, good_header, good_dtype)

        assert len(DATADICT2) == sd2_cols
        assert np.array_equiv(list(DATADICT2.keys()), list(good_header))

        for idx, key in enumerate(DATADICT2):
            assert np.array_equiv(DATADICT2[key], REF_DATA2[idx])
            assert DATADICT2[key][0].dtype == good_dtype


    def test_accuracy_empty(self):
        assert unzip_to_datadict({}, dtype=int) == {}
        assert unzip_to_datadict({}, HEADER=[['a']], dtype=int) == {}
        assert unzip_to_datadict({0:{}}, HEADER=[['a']], dtype=int) == {'a':[]}


class TestUnzipDenseDict:

    @pytest.mark.parametrize('non_sd',
        (0, 1, np.nan, np.pi, [1,2], {1,2}, (1,2), True, False, None,
         min, lambda x: x, np.float64, 'junk', {np.nan, np.nan})
    )
    def test_rejects_non_sparse_dicts(self, non_sd):
        with pytest.raises(TypeError):
            unzip_to_dense_dict(non_sd)


    @pytest.mark.parametrize('bad_sd', ({'a':1}, {0: 'a'}))
    def test_rejects_bad_sparse_dicts(self, bad_sd):
        with pytest.raises(ValueError):
            unzip_to_dense_dict(bad_sd)


    def test_accepts_full_sparse_dict(self, good_sd_1):
        unzip_to_dense_dict(good_sd_1)


    def test_accepts_inner_sparse_dict(self, good_inner_dict):
        unzip_to_dense_dict(good_inner_dict)


    @pytest.mark.parametrize('bad_dtype',
        (0, True, np.nan, np.pi, lambda x: x, {'a': 1}, 'junk', min)
    )
    def test_rejects_bad_dtype(self, bad_dtype, good_sd_1):
        with pytest.raises(TypeError):
            unzip_to_dense_dict(good_sd_1, dtype=bad_dtype)


    @pytest.mark.parametrize('good_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    def test_accepts_good_dtype(self, good_dtype, good_sd_1):
        unzip_to_dense_dict(good_sd_1, dtype=good_dtype)


    @pytest.mark.parametrize('good_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    def test_accuracy(self, good_dtype, good_sd_1, good_inner_dict):

        SD1_DENSE_DICT = {0:{0:1,1:2,2:0}, 1:{0:0,1:1,2:1}}
        INNER_DICT_DENSE_DICT = {0:0, 1:2, 2:0}

        DENSE_DICT1 = unzip_to_dense_dict(good_sd_1, good_dtype)
        assert shape_(DENSE_DICT1) == shape_(good_sd_1)
        assert DENSE_DICT1 == SD1_DENSE_DICT

        if good_dtype in (int, float):
            assert type(DENSE_DICT1[0][0]) == good_dtype
        else:
            assert DENSE_DICT1[0][0].dtype == good_dtype


        DENSE_DICT2 = unzip_to_dense_dict(good_inner_dict, good_dtype)
        assert shape_(DENSE_DICT2) == shape_(good_inner_dict)
        assert DENSE_DICT2 == INNER_DICT_DENSE_DICT

        if good_dtype in (int, float):
            assert type(DENSE_DICT2[0]) == good_dtype
        else:
            assert DENSE_DICT2[0].dtype == good_dtype


    def test_accuracy_empty(self):
        assert unzip_to_dense_dict({}, dtype=int) == {}
        assert unzip_to_dense_dict({0:{}}, dtype=int) == {0:{}}


class TestUnzipPandasDataframe:

    @pytest.mark.parametrize('non_sd',
        (0, 1, np.nan, np.pi, [1,2], {1,2}, (1,2), True, False, None,
         min, lambda x: x, np.float64, 'junk', {np.nan, np.nan})
    )
    def test_rejects_non_sparse_dicts(self, non_sd):
        with pytest.raises(TypeError):
            unzip_to_dataframe(non_sd)


    @pytest.mark.parametrize('bad_sd', ({'a':1}, {0: 'a'}))
    def test_rejects_bad_sparse_dicts(self, bad_sd):
        with pytest.raises(ValueError):
            unzip_to_dataframe(bad_sd)


    def test_accepts_full_sparse_dict(self, good_sd_1):
        unzip_to_dataframe(good_sd_1)


    def test_accepts_inner_sparse_dict(self, good_inner_dict):
        unzip_to_dataframe(good_inner_dict)


    @pytest.mark.parametrize('bad_header',
        (0, True, np.nan, np.pi, lambda x: x, {'a': 1}, 'junk', min, np.float64)
    )
    def test_rejects_bad_header_type(self, bad_header, good_sd_1):
        with pytest.raises(TypeError):
            unzip_to_dataframe(good_sd_1, HEADER=bad_header, dtype=float)


    @pytest.mark.parametrize('bad_header', (list(), list('A'), list('ABCDEFGHI')))
    def test_rejects_bad_header_len(self, bad_header, good_sd_1):
        with pytest.raises(ValueError):
            unzip_to_dataframe(good_sd_1, HEADER=bad_header, dtype=float)


    @pytest.mark.parametrize('good_header',
        (['a', 'b', 'c'], [['a', 'b', 'c']], np.array(['a', 'b', 'c']),
         {'a', 'b', 'c'}, ('a', 'b', 'c'), ['a', 'b'], [['a', 'b']])
    )
    def accepts_good_header(self, good_header, good_sd_1):
        unzip_to_dataframe(good_sd_1, HEADER=good_header, dtype=float)


    @pytest.mark.parametrize('bad_dtype',
        (0, True, np.nan, np.pi, lambda x: x, {'a': 1}, 'junk', min)
    )
    def test_rejects_bad_dtype(self, bad_dtype, good_sd_1):
        with pytest.raises(TypeError):
            unzip_to_dataframe(good_sd_1, dtype=bad_dtype)


    @pytest.mark.parametrize('good_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    def test_accepts_good_dtype(self, good_dtype, good_sd_1):
        unzip_to_dataframe(good_sd_1, dtype=good_dtype)


    @pytest.mark.parametrize('good_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    def test_accuracy_no_header(self, good_dtype, good_sd_1, good_inner_dict,
                      unzipped_good_sd_1, unzipped_good_inner_dict):

        sd1_cols = unzipped_good_sd_1.shape[1]
        REF_DATA1 = pd.DataFrame(
                                 data=unzipped_good_sd_1,
                                 columns=list(map(str, range(sd1_cols))),
                                 dtype=good_dtype
                    )

        DATAFRAME1 = unzip_to_dataframe(good_sd_1, dtype=good_dtype)
        assert DATAFRAME1.equals(REF_DATA1)
        for idx, key in enumerate(DATAFRAME1):
            assert DATAFRAME1[key].dtype == good_dtype


        sd2_cols = 1
        REF_DATA2 = pd.DataFrame(
                                    data=unzipped_good_inner_dict,
                                    columns=list(map(str, range(sd2_cols))),
                                    dtype=good_dtype
                    )

        DATAFRAME2 = unzip_to_dataframe(good_inner_dict, dtype=good_dtype)
        assert DATAFRAME2.equals(REF_DATA2)
        for idx, key in enumerate(DATAFRAME2):
            assert DATAFRAME2[key].dtype == good_dtype


    @pytest.mark.parametrize('good_header',
        (['a', 'b', 'c'], [['a', 'b', 'c']], np.array(['a', 'b', 'c']),
         {'a', 'b', 'c'}, ('a', 'b', 'c'))
    )
    @pytest.mark.parametrize('good_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    def test_accuracy_header1(self, good_header, good_dtype, good_sd_1,
              good_inner_dict, unzipped_good_sd_1, unzipped_good_inner_dict):

        REF_DATA1 = pd.DataFrame(
                                data=unzipped_good_sd_1,
                                columns=np.array(list(good_header)).ravel(),
                                dtype=good_dtype
        )

        DATAFRAME1 = unzip_to_dataframe(good_sd_1, good_header, good_dtype)
        assert DATAFRAME1.equals(REF_DATA1)
        for idx, key in enumerate(DATAFRAME1):
            assert DATAFRAME1[key].dtype == good_dtype


        REF_DATA2 = pd.DataFrame(
                                    data=unzipped_good_inner_dict.reshape((1,-1)),
                                    columns=np.array(list(good_header)).ravel(),
                                    dtype=good_dtype
        )

        DATAFRAME2 = unzip_to_dataframe(good_inner_dict, good_header, good_dtype)
        assert DATAFRAME2.equals(REF_DATA2)
        for idx, key in enumerate(DATAFRAME2):
            assert DATAFRAME2[key].dtype == good_dtype


    @pytest.mark.parametrize('good_header',
        (['a', 'b'], [['a', 'b']], np.array(['a', 'b']), {'a', 'b'}, ('a', 'b'))
    )
    @pytest.mark.parametrize('good_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    def test_accuracy_sd1_header2(self, good_header, good_dtype, good_sd_1,
                                unzipped_good_sd_1):

        REF_DATA1 = pd.DataFrame(
                                data=unzipped_good_sd_1.transpose(),
                                columns=np.array(list(good_header)).ravel(),
                                dtype=good_dtype
        )

        DATAFRAME1 = unzip_to_dataframe(good_sd_1, good_header, good_dtype)
        assert DATAFRAME1.equals(REF_DATA1)

        for idx, key in enumerate(DATAFRAME1):
            assert DATAFRAME1[key].dtype == good_dtype


    @pytest.mark.parametrize('good_header',
        (['a'], [['a']], np.array(['a']), {'a'}, ('a',))
    )
    @pytest.mark.parametrize('good_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    def test_accuracy_sd2_header2(self, good_header, good_dtype, good_inner_dict,
                                  unzipped_good_inner_dict):

        REF_DATA2 = pd.DataFrame(
                                data=unzipped_good_inner_dict,
                                columns=np.array(list(good_header)).ravel(),
                                dtype=good_dtype
        )

        DATAFRAME2 = unzip_to_dataframe(good_inner_dict, good_header, good_dtype)
        assert DATAFRAME2.equals(REF_DATA2)

        for idx, key in enumerate(DATAFRAME2):
            assert DATAFRAME2[key].dtype == good_dtype


    def test_accuracy_empty(self):
        assert unzip_to_dataframe({}, dtype=int).equals(pd.DataFrame({}, dtype=int))
        assert unzip_to_dataframe({0:{}}, HEADER=[['a']], dtype=int).equals(
            pd.DataFrame({'a':[]}, dtype=int))
        assert unzip_to_dataframe({0:{}}, HEADER=[['a']], dtype=np.uint8).equals(
            pd.DataFrame({'a':[]}, dtype=np.uint8))


class TestUnzipDaskDataframe:

    @pytest.mark.parametrize('non_sd',
        (0, 1, np.nan, np.pi, [1,2], {1,2}, (1,2), True, False, None,
         min, lambda x: x, np.float64, 'junk', {np.nan, np.nan})
    )
    def test_rejects_non_sparse_dicts(self, non_sd):
        with pytest.raises(TypeError):
            unzip_to_dask_dataframe(non_sd)


    @pytest.mark.parametrize('bad_sd', ({'a':1}, {0: 'a'}))
    def test_rejects_bad_sparse_dicts(self, bad_sd):
        with pytest.raises(ValueError):
            unzip_to_dask_dataframe(bad_sd)


    def test_accepts_full_sparse_dict(self, good_sd_1):
        unzip_to_dask_dataframe(good_sd_1)


    def test_accepts_inner_sparse_dict(self, good_inner_dict):
        unzip_to_dask_dataframe(good_inner_dict)


    @pytest.mark.parametrize('bad_header',
        (0, True, np.nan, np.pi, lambda x: x, {'a': 1}, 'junk', min, np.float64)
    )
    def test_rejects_bad_header_type(self, bad_header, good_sd_1):
        with pytest.raises(TypeError):
            unzip_to_dask_dataframe(good_sd_1, HEADER=bad_header, dtype=float)


    @pytest.mark.parametrize('bad_header', (list(), list('A'), list('ABCDEFGHI')))
    def test_rejects_bad_header_len(self, bad_header, good_sd_1):
        with pytest.raises(ValueError):
            unzip_to_dask_dataframe(good_sd_1, HEADER=bad_header, dtype=float)


    @pytest.mark.parametrize('good_header',
        (['a', 'b', 'c'], [['a', 'b', 'c']], np.array(['a', 'b', 'c']),
         {'a', 'b', 'c'}, ('a', 'b', 'c'), ['a', 'b'], [['a', 'b']])
    )
    def accepts_good_header(self, good_header, good_sd_1):
        unzip_to_dask_dataframe(good_sd_1, HEADER=good_header, dtype=float)


    @pytest.mark.parametrize('bad_dtype',
        (0, True, np.nan, np.pi, lambda x: x, {'a': 1}, 'junk', min)
    )
    def test_rejects_bad_dtype(self, bad_dtype, good_sd_1):
        with pytest.raises(TypeError):
            unzip_to_dask_dataframe(good_sd_1, dtype=bad_dtype)


    @pytest.mark.parametrize('good_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    def test_accepts_good_dtype(self, good_dtype, good_sd_1):
        unzip_to_dask_dataframe(good_sd_1, dtype=good_dtype)


    @pytest.mark.parametrize('junk_chunksize_type',
        (None, 'junk', {'a':1}, lambda x: x, True, [[1,2]], {1,2})
    )
    def test_rejects_junk_chunksize(self, good_sd_1, junk_chunksize_type):
        with pytest.raises(TypeError):
            unzip_to_dask_dataframe(good_sd_1, chunksize=junk_chunksize_type)

    @pytest.mark.parametrize('bad_chunksize',
                             ((), (0, ), (0, 0), 0, -10, np.pi, np.nan)
    )
    def test_rejects_bad_chunksize(self, good_sd_1, bad_chunksize):
        with pytest.raises(ValueError):
            unzip_to_dask_dataframe(good_sd_1, chunksize=bad_chunksize)


    @pytest.mark.parametrize('good_chunksize',
                 ([1,2], [1,], (1,2,3), (3, ), 1, 40_000)
    )
    def test_accepts_good_chunksize(self, good_sd_1, good_chunksize):
        unzip_to_dask_dataframe(good_sd_1, chunksize=good_chunksize)


    @pytest.mark.parametrize('good_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    @pytest.mark.parametrize('good_chunksize_sd, good_chunksize_ddf',
        ([2,2], [(2,3), 2], [(1,3), 1])
    )
    def test_accuracy_no_header(self, good_dtype, good_sd_1, good_inner_dict,
              unzipped_good_sd_1, unzipped_good_inner_dict, good_chunksize_sd,
              good_chunksize_ddf):

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        sd1_cols = unzipped_good_sd_1.shape[1]
        REF_DATA1 = ddf.from_pandas(
                                    pd.DataFrame(
                                        data=unzipped_good_sd_1,
                                        columns=list(map(str, range(sd1_cols))),
                                        dtype=good_dtype
                                    ),
                                    chunksize=good_chunksize_ddf
        )

        DDF1 = unzip_to_dask_dataframe(good_sd_1, chunksize=good_chunksize_sd,
                                       dtype=good_dtype)

        assert DDF1.compute().equals(REF_DATA1.compute())

        for idx, key in enumerate(DDF1):
            assert DDF1[key].dtype == good_dtype

        assert DDF1.get_partition(0).compute().shape == \
                    REF_DATA1.get_partition(0).compute().shape

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        sd2_cols = 1
        REF_DATA2 = ddf.from_pandas(
                            pd.DataFrame(
                                        data=unzipped_good_inner_dict,
                                        columns=list(map(str, range(sd2_cols))),
                                        dtype=good_dtype
                            ),
                            chunksize=good_chunksize_ddf
        )
        DDF2 = unzip_to_dask_dataframe(
                                        good_inner_dict,
                                        chunksize=good_chunksize_sd,
                                        dtype=good_dtype
        )

        assert DDF2.compute().equals(REF_DATA2.compute())

        for idx, key in enumerate(DDF2):
            assert DDF2[key].dtype == good_dtype

        assert DDF2.get_partition(0).compute().shape == \
                    REF_DATA2.get_partition(0).compute().shape

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @pytest.mark.parametrize('good_header',
        (['a', 'b', 'c'], [['a', 'b', 'c']], np.array(['a', 'b', 'c']),
         {'a', 'b', 'c'}, ('a', 'b', 'c'))
    )
    @pytest.mark.parametrize('good_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    @pytest.mark.parametrize('good_chunksize_sd, good_chunksize_ddf',
        ([2,2], [(2,2), 2], [(1,3), 1])
    )
    def test_accuracy_header1(self, good_header, good_dtype, good_sd_1,
          good_inner_dict, unzipped_good_sd_1, unzipped_good_inner_dict,
          good_chunksize_sd, good_chunksize_ddf):

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        REF_DATA1 = ddf.from_pandas(
                    pd.DataFrame(
                                 data=unzipped_good_sd_1,
                                 columns=np.array(list(good_header)).ravel(),
                                 dtype=good_dtype
                    ),
                    chunksize=good_chunksize_ddf
        )

        DDF1 = unzip_to_dask_dataframe(good_sd_1, chunksize=good_chunksize_sd,
                                    HEADER=good_header, dtype=good_dtype)

        assert DDF1.compute().equals(REF_DATA1.compute())

        for idx, key in enumerate(DDF1):
            assert DDF1[key].dtype == good_dtype

        assert DDF1.get_partition(0).compute().shape == \
                    REF_DATA1.get_partition(0).compute().shape

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        REF_DATA2 = ddf.from_pandas(
                pd.DataFrame(
                                data=unzipped_good_inner_dict.reshape((1,-1)),
                                columns=np.array(list(good_header)).ravel(),
                                dtype=good_dtype
                ),
                chunksize=good_chunksize_ddf
        )

        DDF2 = unzip_to_dask_dataframe(
                                       good_inner_dict,
                                       chunksize=good_chunksize_sd,
                                       HEADER=good_header,
                                       dtype=good_dtype
        )
        assert DDF2.compute().equals(REF_DATA2.compute())
        for idx, key in enumerate(DDF2):
            assert DDF2[key].dtype == good_dtype

        assert DDF2.get_partition(0).compute().shape == \
                    REF_DATA2.get_partition(0).compute().shape

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @pytest.mark.parametrize('good_header',
        (['a', 'b'], [['a', 'b']], np.array(['a', 'b']), {'a', 'b'}, ('a', 'b'))
    )
    @pytest.mark.parametrize('good_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    @pytest.mark.parametrize('good_chunksize_sd, good_chunksize_ddf',
        ([2,2], [(2,2), 2], [(1,3), 1])
    )
    def test_accuracy_sd1_header2(self, good_header, good_dtype, good_sd_1,
                unzipped_good_sd_1, good_chunksize_sd, good_chunksize_ddf):

        REF_DATA1 = ddf.from_pandas(
                    pd.DataFrame(
                                data=unzipped_good_sd_1.transpose(),
                                columns=np.array(list(good_header)).ravel(),
                                dtype=good_dtype
                    ),
                    chunksize=good_chunksize_ddf
        )
        DDF1 = unzip_to_dask_dataframe(
                                        good_sd_1,
                                        chunksize=good_chunksize_sd,
                                        HEADER=good_header,
                                        dtype=good_dtype
        )

        assert DDF1.compute().equals(REF_DATA1.compute())

        for idx, key in enumerate(DDF1):
            assert DDF1[key].dtype == good_dtype

        assert DDF1.get_partition(0).compute().shape == \
                    REF_DATA1.get_partition(0).compute().shape


    @pytest.mark.parametrize('good_header',
        (['a'], [['a']], np.array(['a']), {'a'}, ('a',))
    )
    @pytest.mark.parametrize('good_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    @pytest.mark.parametrize('good_chunksize_sd, good_chunksize_ddf',
        ([2,2], [(2,2), 2], [(1,3), 1], (100, 100))
    )
    def test_accuracy_sd2_header2(self, good_header, good_dtype, good_inner_dict,
          unzipped_good_inner_dict, good_chunksize_sd, good_chunksize_ddf):

        REF_DATA2 = ddf.from_pandas(
                    pd.DataFrame(
                                data=unzipped_good_inner_dict,
                                columns=np.array(list(good_header)).ravel(),
                                dtype=good_dtype
                    ),
                    chunksize=good_chunksize_ddf
        )

        DDF2 = unzip_to_dask_dataframe(
                                       good_inner_dict,
                                       chunksize=good_chunksize_sd,
                                       HEADER=good_header,
                                       dtype=good_dtype
        )
        assert DDF2.compute().equals(REF_DATA2.compute())

        for idx, key in enumerate(DDF2):
            assert DDF2[key].dtype == good_dtype

        assert DDF2.get_partition(0).compute().shape == \
                    REF_DATA2.get_partition(0).compute().shape


    def test_accuracy_empty(self):

        empty_dask_df1 = ddf.from_pandas(
                                         pd.DataFrame({'a':[]}, dtype=int),
                                         npartitions=1
        )

        DASK_TEST_1 = unzip_to_dask_dataframe({}, dtype=int)
        DASK_TEST_2 = unzip_to_dask_dataframe({0:{}}, HEADER=[['a']], dtype=int)

        assert DASK_TEST_1.compute().equals(pd.DataFrame({}, dtype=int))
        assert DASK_TEST_2.compute().equals(empty_dask_df1.compute().astype(int))
        assert unzip_to_dask_dataframe({0:{}}, HEADER=[['a']], dtype=np.uint8
                ).compute().equals(empty_dask_df1.astype(np.uint8).compute())






















