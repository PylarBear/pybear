# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

bypass = True


import pytest
from sparse_dict._transform import (
                                    zip_ndarray,
                                    zip_dask_array,
                                    zip_datadict,
                                    zip_dataframe,
                                    zip_dask_dataframe,

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


@pytest.mark.skipif(bypass is True, reason=f'')
class TestZipArray:

    @pytest.mark.parametrize('junk',
        ('abc', 0, False, None, np.nan, min, lambda x: x, {'a':1})
    )
    def test_rejects_non_list(self, junk):
        with pytest.raises(TypeError):
            zip_ndarray(junk, dtype=float)


    def test_rejects_dask_array(self):
        with pytest.raises(TypeError):
            zip_ndarray(da.array([1,2,3]), dtype=float)


    def test_rejects_dataframes(self):
        with pytest.raises(TypeError):
            zip_ndarray(pd.DataFrame({'a':[1,2], 'b':[3,4]}), dtype=float)

        with pytest.raises(TypeError):
            zip_ndarray(ddf.from_array(np.random.randint(0,10,(3,3))), dtype=float)


    def test_rejects_series(self):
        with pytest.raises(TypeError):
            zip_ndarray(pd.Series([1,2]), dtype=float)

        with pytest.raises(TypeError):
            zip_ndarray(ddf.from_array(np.array([1,2,3])), dtype=float)


    def test_rejects_sparse_dicts(self):
        with pytest.raises(TypeError):
            zip_ndarray({0:{0:1,1:2}, 1:{0:2,1:3}}, dtype=float)


    @pytest.mark.parametrize('array_like',
        ([1,2,3], (1,2,3), {1,2,3}, np.random.randint(1,10,(3,)))
    )
    def test_accepts_1D_array_like(self, array_like):
        zip_ndarray(array_like, dtype=float)


    @pytest.mark.parametrize('array_like',
        ([[1,2],[3,4]], np.random.randint(1,10,(3,3)))
    )
    def test_accepts_2D_array_like(self, array_like):
        zip_ndarray(array_like, dtype=float)


    def test_rejects_3D_np(self):
        with pytest.raises(ValueError):
            zip_ndarray(np.random.randint(1,10,(3,3,3)), dtype=float)


    def test_rejects_3D_list(self):
        with pytest.raises(ValueError):
            ARRAY = [
                [
                    [0, 1],
                    [2, 3]
                ],
                [
                    [4, 5],
                    [6, 7]
                ]
            ]

            zip_ndarray(ARRAY, dtype=float)


    @pytest.mark.parametrize('_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    def test_correct_dtypes(self, _dtype):
        OUTPUT = zip_ndarray(np.random.randint(1,10,(2,2)), dtype=_dtype)

        if _dtype == int:
            assert type(OUTPUT[0][0]) is int
        elif _dtype == float:
            assert type(OUTPUT[0][0]) is float
        else:
            assert OUTPUT[0][0].dtype == _dtype


    def test_accepts_ragged_array(self):
        OUTPUT = zip_ndarray([[1,2],[4],[6,7,8]], dtype=float)
        assert OUTPUT == {0:{0:1,1:2}, 1:{0:4}, 2:{0:6,1:7,2:8}}


    def test_accuracy_zeros(self):
        OUTPUT = zip_ndarray([0,0,0], dtype=int)
        assert OUTPUT == {2:0}


    def test_accuracy_non_zeros(self):
        OUTPUT = zip_ndarray([1,1,1], dtype=int)
        assert OUTPUT == {0:1, 1:1, 2:1}


    def test_accuracy_inner(self):
        assert zip_ndarray([1,0,8], dtype=int) == {0:1, 2:8}


    def test_accuracy_outer(self):
        assert zip_ndarray([[1,2], [0,4]], dtype=int) == {0:{0:1,1:2}, 1:{1:4}}


@pytest.mark.skipif(bypass is True, reason=f'')
class TestZipDaskArray:

    @pytest.mark.parametrize('junk',
        ('abc', 0, False, None, np.nan, min, lambda x: x, {'a':1},
         pd.DataFrame({'a': [1, 2], 'b': [3, 4]}),
         ddf.from_array(np.random.randint(0, 10, (3, 3))),
         pd.Series([1, 2]),
         ddf.from_array(np.array([1, 2, 3])),
         {0:{0:1, 1:2}, 1:{0:2, 1:3}}
         )
    )
    def test_rejects_anything_not_dask_array(self, junk):
        with pytest.raises(TypeError):
            zip_dask_array(junk, dtype=float)


    def test_accepts_1D_dask_array(self):
        zip_dask_array(
                        da.random.randint(0,1,(10,), chunks=(5,)),
                        dtype=float
        )


    def test_accepts_2D_dask_array(self):
        zip_dask_array(
                        da.random.randint(0,1,(10,10), chunks=(5,5)),
                        dtype=float
        )


    def test_rejects_3D_dask_array(self):
        with pytest.raises(ValueError):
            zip_dask_array(
                            da.random.randint(0,1,(4,4,4), chunks=(2,2,2)),
                            dtype=float
            )

    @pytest.mark.parametrize('_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    def test_correct_dtypes(self, _dtype):
        OUTPUT = zip_dask_array(
                                da.random.randint(1,10,(4,4), chunks=(2,2)),
                                dtype=_dtype
        )

        if _dtype == int:
            assert type(OUTPUT[0][0]) is int
        elif _dtype == float:
            assert type(OUTPUT[0][0]) is float
        else:
            assert OUTPUT[0][0].dtype == _dtype


    def test_accuracy_zeros(self):
        OUTPUT = zip_dask_array(da.array([0,0,0]), dtype=int)
        assert OUTPUT == {2:0}


    def test_accuracy_non_zeros(self):
        OUTPUT = zip_dask_array(da.array([1,1,1]), dtype=int)
        assert OUTPUT == {0:1, 1:1, 2:1}


    def test_accuracy_inner(self):
        OUTPUT = zip_dask_array(da.array([1,0,8], dtype=int))
        assert OUTPUT == {0:1, 2:8}


    def test_accuracy_outer(self):
        OUTPUT = zip_dask_array(da.array([[1,2], [0,4]]),dtype=int)
        assert OUTPUT == {0:{0:1,1:2}, 1:{1:4}}


    @pytest.mark.parametrize('chunk_size',
                             ((4,8), (4,4), (4,2), (4,1), (8,8)),
    )
    def test_accepts_any_valid_chunking(self, chunk_size):

        DA = da.random.randint(0,1,(8,8), chunks=chunk_size)
        NP = DA.compute()
        NP_RESULT = zip_ndarray(NP)
        DA_RESULT = zip_dask_array(DA)
        assert DA_RESULT == NP_RESULT


    def test_handles_a_really_big_dask_array(self):

        DA = da.random.randint(0,1,(10_000,10_000), chunks=(1_000,10_000))
        SD = zip_dask_array(DA)

        assert DA.shape == shape_(SD)



@pytest.mark.skipif(bypass is True, reason=f'')
class TestZipDatadict:

    @staticmethod
    @pytest.fixture
    def good_datadict():
        raw_data = np.random.randint(1,10,(5,3)).transpose()
        columns = list('ABC')
        return dict((zip(columns, raw_data)))


    @pytest.mark.parametrize('junk',
        ('abc', 0, False, None, np.nan, min, lambda x: x, {'a':1}, [1,2,3],
         {1,2,3}, (1,2,3))
    )
    def test_rejects_junk(self, junk):
        with pytest.raises(TypeError):
            zip_datadict(junk, dtype=float)

    def test_rejects_dask_array(self):
        with pytest.raises(TypeError):
            zip_datadict(da.array([1, 2, 3]), dtype=float)

    def test_rejects_dataframes(self):
        with pytest.raises(TypeError):
            zip_datadict(pd.DataFrame({'a': [1, 2], 'b': [3, 4]}), dtype=float)

        with pytest.raises(TypeError):
            zip_datadict(ddf.from_array(np.random.randint(0, 10, (3, 3))),
                         dtype=float
            )

    def test_rejects_series(self):
        with pytest.raises(TypeError):
            zip_datadict(pd.Series([1, 2]), dtype=float)

        with pytest.raises(TypeError):
            zip_datadict(ddf.from_array(np.array([1, 2, 3])), dtype=float)

    def test_rejects_sparse_dicts(self):
        with pytest.raises(TypeError):
            zip_datadict({0: {0: 1, 1: 2}, 1: {0: 2, 1: 3}}, dtype=float)


    def test_accepts_datadict(self):
        zip_datadict({'A': [1,2,3], 'B':[7,8,9]}, dtype=float)


    @pytest.mark.parametrize('_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    def test_correct_dtypes(self, _dtype, good_datadict):
        SD, HEADER = zip_datadict(good_datadict, dtype=_dtype)

        if _dtype == int:
            assert type(SD[0][0]) is int
        elif _dtype == float:
            assert type(SD[0][0]) is float
        else:
            assert SD[0][0].dtype == _dtype

        assert np.array_equiv(HEADER, ['A', 'B', 'C'])
        assert shape_(SD) == (5, 3)


    def test_accuracy_zeros(self):
        SD, HEADER = zip_datadict({'a':[0,0,0], 'b':[0,0,0]}, dtype=int)
        assert SD == {0:{1:0}, 1:{1:0}, 2:{1:0}}
        assert np.array_equiv(HEADER, ['a', 'b'])


    def test_accuracy_non_zeros(self):
        SD, HEADER = zip_datadict({'a':[1, 1], 'b':[2, 2]}, dtype=int)
        assert SD == {0: {0: 1, 1: 2}, 1:{0:1, 1: 2}}
        assert np.array_equiv(HEADER, ['a', 'b'])


@pytest.mark.skipif(bypass is True, reason=f'')
class TestZipPandasDataFrame:

    @pytest.mark.parametrize('junk',
        ('abc', 0, False, None, np.nan, min, lambda x: x, {'a':1}, [1,2,3],
         {1,2,3}, (1,2,3), ddf.from_array(np.random.randint(0,10,(3,3))),
         ddf.from_array(np.array([1, 2, 3])))
    )
    def test_rejects_non_pandas_df(self, junk):
        with pytest.raises(TypeError):
            zip_dataframe(junk, dtype=float)


    def test_accepts_pandas_dataframe(self):
        zip_dataframe(pd.DataFrame({'a':[1,2], 'b':[3,4]}), dtype=float)


    def test_accepts_series(self):
        with pytest.raises(TypeError):
            zip_dataframe(pd.Series([1,2]), dtype=float)


    def test_rejects_sparse_dicts(self):
        with pytest.raises(TypeError):
            zip_dataframe({0:{0:1,1:2}, 1:{0:2,1:3}}, dtype=float)


    @pytest.mark.parametrize('_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    def test_correct_dtypes(self, _dtype):
        SD, HEADER = zip_dataframe(
            pd.DataFrame(data=np.random.randint(1,10,(2,2)), columns=list('AB')),
            dtype=_dtype
        )

        if _dtype == int:
            assert type(SD[0][0]) is int
        elif _dtype == float:
            assert type(SD[0][0]) is float
        else:
            assert SD[0][0].dtype == _dtype

        assert np.array_equiv(HEADER, ['A', 'B'])


    def test_accuracy_zeros(self):
        SD, HEADER = zip_dataframe(pd.DataFrame({'A': [0,0,0]}), dtype=int)
        assert SD == {0: {2:0}}
        assert np.array_equiv(HEADER, ['A'])


    def test_accuracy_non_zeros(self):
        SD, HEADER = zip_dataframe(pd.DataFrame({'A':[1,1,1]}), dtype=int)
        assert SD == {0: {0:1, 1:1, 2:1}}
        assert np.array_equiv(HEADER, ['A'])


@pytest.mark.skipif(bypass is True, reason=f'')
class TestZipDaskDataFrame:

    @pytest.mark.parametrize('junk',
        ('abc', 0, False, None, np.nan, min, lambda x: x, {'a':1}, [1,2,3],
         {1,2,3}, (1,2,3), pd.DataFrame({'a':[1,2], 'b':[3,4]}), pd.Series([1,2])
         )
    )
    def test_rejects_non_dask_df(self, junk):
        with pytest.raises(TypeError):
            zip_dask_dataframe(junk, dtype=float)


    def test_accepts_dask_dataframes(self):
        zip_dask_dataframe(
                            ddf.from_array(np.random.randint(0,10,(3,3))),
                            dtype=float
        )


    def test_accepts_dask_series(self):

        with pytest.raises(TypeError):
            zip_dask_dataframe(ddf.from_array(np.array([1,2,3])), dtype=float)


    def test_rejects_sparse_dicts(self):
        with pytest.raises(TypeError):
            zip_dask_dataframe({0:{0:1,1:2}, 1:{0:2,1:3}}, dtype=float)


    @pytest.mark.parametrize('_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    def test_correct_dtypes(self, _dtype):
        SD, HEADER = zip_dask_dataframe(
            ddf.from_pandas(
                pd.DataFrame(data=np.random.randint(1,10,(2,2)),
                             columns=['A', 'B'])
            ),
            dtype=_dtype
        )

        if _dtype == int:
            assert type(SD[0][0]) is int
        elif _dtype == float:
            assert type(SD[0][0]) is float
        else:
            assert SD[0][0].dtype == _dtype

        assert np.array_equiv(HEADER, ['A', 'B'])


    def test_accuracy_zeros(self):
        SD, HEADER = zip_dask_dataframe(
            ddf.from_pandas(
                pd.DataFrame({'A': [0,0,0]})
            ),
                dtype=int
        )
        assert SD == {0: {2:0}}
        assert np.array_equiv(HEADER, ['A'])


    def test_accuracy_non_zeros(self):
        SD, HEADER = zip_dask_dataframe(
            ddf.from_pandas(
                pd.DataFrame({'A': [1,1,1]})
            ),
            dtype=int
        )
        assert SD == {0: {0:1, 1:1, 2:1}}
        assert np.array_equiv(HEADER, ['A'])




# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
# ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


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


@pytest.mark.skipif(bypass is True, reason=f'')
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


    @pytest.mark.parametrize('bad_header',
        (0, True, np.nan, np.pi, lambda x: x, {'a': 1}, 'junk', min, np.float64)
    )
    def test_rejects_bad_header_type(self, bad_header, good_sd_1):
        with pytest.raises(TypeError):
            unzip_to_ndarray(good_sd_1, HEADER=bad_header, dtype=float)


    @pytest.mark.parametrize('bad_header', (list(), list('A'), list('ABCDEFGHI')))
    def test_rejects_bad_header_len(self, bad_header, good_sd_1):
        with pytest.raises(ValueError):
            unzip_to_ndarray(good_sd_1, HEADER=bad_header, dtype=float)


    @pytest.mark.parametrize('good_header',
        (['a', 'b', 'c'], [['a', 'b', 'c']], np.array(['a', 'b', 'c']),
         {'a', 'b', 'c'}, ('a', 'b', 'c'), ['a', 'b'], [['a', 'b']])
    )
    def accepts_good_header(self, good_header, good_sd_1):
        unzip_to_ndarray(good_sd_1, HEADER=good_header, dtype=float)


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


    @pytest.mark.parametrize('empty_sd', ({}, {0:{}}))
    def test_empty_handling(self, empty_sd):
        with pytest.raises(ValueError):
            unzip_to_ndarray(empty_sd)


    @pytest.mark.parametrize('good_header',
        (['a', 'b', 'c'], [['a', 'b', 'c']], np.array(['a', 'b', 'c']),
         {'a', 'b', 'c'}, ('a', 'b', 'c'))
    )
    @pytest.mark.parametrize('good_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    def test_accuracy(self, good_header, good_dtype, good_sd_1, good_inner_dict,
                      unzipped_good_sd_1, unzipped_good_inner_dict):
        NDARRAY1, HEADER = unzip_to_ndarray(good_sd_1, good_header, good_dtype)
        assert NDARRAY1.shape == shape_(good_sd_1)
        assert np.array_equiv(NDARRAY1, unzipped_good_sd_1.astype(good_dtype))
        assert np.array_equiv(HEADER, np.array(list(good_header)).ravel())
        assert NDARRAY1.dtype == good_dtype

        NDARRAY2, HEADER = unzip_to_ndarray(good_inner_dict, good_header, good_dtype)
        assert NDARRAY2.shape == shape_(good_inner_dict)
        assert np.array_equiv(NDARRAY2, unzipped_good_inner_dict.astype(good_dtype))
        assert np.array_equiv(HEADER, np.array(list(good_header)).ravel())
        assert NDARRAY2.dtype == good_dtype


@pytest.mark.skipif(bypass is True, reason=f'')
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


    @pytest.mark.parametrize('bad_header',
        (0, True, np.nan, np.pi, lambda x: x, {'a': 1}, 'junk', min, np.float64)
    )
    def test_rejects_bad_header_type(self, bad_header, good_sd_1):
        with pytest.raises(TypeError):
            unzip_to_list(good_sd_1, HEADER=bad_header, dtype=float)


    @pytest.mark.parametrize('bad_header', (list(), list('A'), list('ABCDEFGHI')))
    def test_rejects_bad_header_len(self, bad_header, good_sd_1):
        with pytest.raises(ValueError):
            unzip_to_list(good_sd_1, HEADER=bad_header, dtype=float)


    @pytest.mark.parametrize('good_header',
        (['a', 'b', 'c'], [['a', 'b', 'c']], np.array(['a', 'b', 'c']),
         {'a', 'b', 'c'}, ('a', 'b', 'c'), ['a', 'b'], [['a', 'b']])
    )
    def accepts_good_header(self, good_header, good_sd_1):
        unzip_to_list(good_sd_1, HEADER=good_header, dtype=float)


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


    @pytest.mark.parametrize('empty_sd', ({}, {0:{}}))
    def test_empty_handling(self, empty_sd):
        with pytest.raises(ValueError):
            unzip_to_list(empty_sd)


    @pytest.mark.parametrize('good_header',
        (['a', 'b', 'c'], [['a', 'b', 'c']], np.array(['a', 'b', 'c']),
         {'a', 'b', 'c'}, ('a', 'b', 'c'))
    )
    @pytest.mark.parametrize('good_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    def test_accuracy(self, good_header, good_dtype, good_sd_1, good_inner_dict,
                      unzipped_good_sd_1, unzipped_good_inner_dict):
        NDARRAY1, HEADER = unzip_to_list(good_sd_1, good_header, good_dtype)
        assert len(NDARRAY1) == outer_len(good_sd_1)
        assert len(NDARRAY1[0]) == inner_len(good_sd_1)
        assert np.array_equiv(NDARRAY1, unzipped_good_sd_1.astype(good_dtype))
        assert np.array_equiv(HEADER, np.array(list(good_header)).ravel())
        assert str(good_dtype).upper() in str(type(NDARRAY1[0][0])).upper()

        NDARRAY2, HEADER = unzip_to_list(good_inner_dict, good_header, good_dtype)
        assert len(NDARRAY2) == inner_len(good_inner_dict)
        assert np.array_equiv(NDARRAY2, unzipped_good_inner_dict.astype(good_dtype))
        assert np.array_equiv(HEADER, np.array(list(good_header)).ravel())
        assert str(good_dtype).upper() in str(type(NDARRAY2[0])).upper()


@pytest.mark.skipif(bypass is True, reason=f'')
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


    @pytest.mark.parametrize('bad_header',
        (0, True, np.nan, np.pi, lambda x: x, {'a': 1}, 'junk', min, np.float64)
    )
    def test_rejects_bad_header_type(self, bad_header, good_sd_1):
        with pytest.raises(TypeError):
            unzip_to_dask_array(good_sd_1, HEADER=bad_header, dtype=float)


    @pytest.mark.parametrize('bad_header', (list(), list('A'), list('ABCDEFGHI')))
    def test_rejects_bad_header_len(self, bad_header, good_sd_1):
        with pytest.raises(ValueError):
            unzip_to_dask_array(good_sd_1, HEADER=bad_header, dtype=float)


    @pytest.mark.parametrize('good_header',
        (['a', 'b', 'c'], [['a', 'b', 'c']], np.array(['a', 'b', 'c']),
         {'a', 'b', 'c'}, ('a', 'b', 'c'), ['a', 'b'], [['a', 'b']])
    )
    def accepts_good_header(self, good_header, good_sd_1):
        unzip_to_dask_array(good_sd_1, HEADER=good_header, dtype=float)


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

    @pytest.mark.parametrize('empty_sd', ({}, {0:{}}))
    def test_empty_handling(self, empty_sd):
        with pytest.raises(ValueError):
            unzip_to_dask_array(empty_sd)


    @pytest.mark.parametrize('good_header',
        (['a', 'b', 'c'], [['a', 'b', 'c']], np.array(['a', 'b', 'c']),
         {'a', 'b', 'c'}, ('a', 'b', 'c'))
    )
    @pytest.mark.parametrize('good_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    def test_accuracy(self, good_header, good_dtype, good_sd_1, good_inner_dict,
                      unzipped_good_sd_1, unzipped_good_inner_dict):
        DASK_ARRAY1, HEADER = \
            unzip_to_dask_array(good_sd_1, good_header, good_dtype, chunks=(1,3))
        assert DASK_ARRAY1.shape == shape_(good_sd_1)
        assert np.array_equiv(DASK_ARRAY1.compute(), unzipped_good_sd_1)
        assert np.array_equiv(HEADER, np.array(list(good_header)).ravel())
        assert DASK_ARRAY1.dtype == good_dtype
        assert DASK_ARRAY1.chunks == ((1, 1), (3,))

        DASK_ARRAY2, HEADER = \
            unzip_to_dask_array(good_inner_dict, good_header, good_dtype, chunks=(2,))
        assert DASK_ARRAY2.shape == shape_(good_inner_dict)
        assert np.array_equiv(DASK_ARRAY2.compute(), unzipped_good_inner_dict)
        assert np.array_equiv(HEADER, np.array(list(good_header)).ravel())
        assert DASK_ARRAY2.dtype == good_dtype
        assert DASK_ARRAY2.chunks == ((2,1),)





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


    @pytest.mark.parametrize('junk_chunk',
        ([1,2], {1,2}, [1,], 1, 'junk', {'a':1}, lambda x: x, 0, True, np.pi)
    )
    def test_rejects_junk_chunks(self, good_sd_1, junk_chunk):
        with pytest.raises(TypeError):
            unzip_to_datadict(good_sd_1, chunks=junk_chunk)

    @pytest.mark.parametrize('bad_chunk',
                             ((), (1,2,3), (1,2,3,4), (0, ), (0, 0))
    )
    def test_rejects_bad_chunks(self, good_sd_1, bad_chunk):
        with pytest.raises(ValueError):
            unzip_to_datadict(good_sd_1, chunks=bad_chunk)


    @pytest.mark.parametrize('empty_sd', ({}, {0:{}}))
    def test_empty_handling(self, empty_sd):
        with pytest.raises(ValueError):
            unzip_to_datadict(empty_sd)


    @pytest.mark.parametrize('good_header',
        (['a', 'b', 'c'], [['a', 'b', 'c']], np.array(['a', 'b', 'c']),
         {'a', 'b', 'c'}, ('a', 'b', 'c'))
    )
    @pytest.mark.parametrize('good_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    def test_accuracy(self, good_header, good_dtype, good_sd_1, good_inner_dict,
                      unzipped_good_sd_1, unzipped_good_inner_dict):
        DASK_ARRAY1, HEADER = \
            unzip_to_datadict(good_sd_1, good_header, good_dtype, chunks=(1,3))
        assert DASK_ARRAY1.shape == shape_(good_sd_1)
        assert np.array_equiv(DASK_ARRAY1.compute(), unzipped_good_sd_1)
        assert np.array_equiv(HEADER, np.array(list(good_header)).ravel())
        assert DASK_ARRAY1.dtype == good_dtype
        assert DASK_ARRAY1.chunks == ((1, 1), (3,))

        DASK_ARRAY2, HEADER = \
            unzip_to_dask_array(good_inner_dict, good_header, good_dtype, chunks=(2,))
        assert DASK_ARRAY2.shape == shape_(good_inner_dict)
        assert np.array_equiv(DASK_ARRAY2.compute(), unzipped_good_inner_dict)
        assert np.array_equiv(HEADER, np.array(list(good_header)).ravel())
        assert DASK_ARRAY2.dtype == good_dtype
        assert DASK_ARRAY2.chunks == ((2,1),)



@pytest.mark.skip
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


    @pytest.mark.parametrize('bad_header',
        (0, True, np.nan, np.pi, lambda x: x, {'a': 1}, 'junk', min, np.float64)
    )
    def test_rejects_bad_header_type(self, bad_header, good_sd_1):
        with pytest.raises(TypeError):
            unzip_to_dense_dict(good_sd_1, HEADER=bad_header, dtype=float)


    @pytest.mark.parametrize('bad_header', (list(), list('A'), list('ABCDEFGHI')))
    def test_rejects_bad_header_len(self, bad_header, good_sd_1):
        with pytest.raises(ValueError):
            unzip_to_dense_dict(good_sd_1, HEADER=bad_header, dtype=float)


    @pytest.mark.parametrize('good_header',
        (['a', 'b', 'c'], [['a', 'b', 'c']], np.array(['a', 'b', 'c']),
         {'a', 'b', 'c'}, ('a', 'b', 'c'), ['a', 'b'], [['a', 'b']])
    )
    def accepts_good_header(self, good_header, good_sd_1):
        unzip_to_dense_dict(good_sd_1, HEADER=good_header, dtype=float)


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


    @pytest.mark.parametrize('junk_chunk',
        ([1,2], {1,2}, [1,], 1, 'junk', {'a':1}, lambda x: x, 0, True, np.pi)
    )
    def test_rejects_junk_chunks(self, good_sd_1, junk_chunk):
        with pytest.raises(TypeError):
            unzip_to_dense_dict(good_sd_1, chunks=junk_chunk)

    @pytest.mark.parametrize('bad_chunk',
                             ((), (1,2,3), (1,2,3,4), (0, ), (0, 0))
    )
    def test_rejects_bad_chunks(self, good_sd_1, bad_chunk):
        with pytest.raises(ValueError):
            unzip_to_dense_dict(good_sd_1, chunks=bad_chunk)

    @pytest.mark.parametrize('empty_sd', ({}, {0:{}}))
    def test_empty_handling(self, empty_sd):
        with pytest.raises(ValueError):
            unzip_to_dense_dict(empty_sd)


    @pytest.mark.parametrize('good_header',
        (['a', 'b', 'c'], [['a', 'b', 'c']], np.array(['a', 'b', 'c']),
         {'a', 'b', 'c'}, ('a', 'b', 'c'))
    )
    @pytest.mark.parametrize('good_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    def test_accuracy(self, good_header, good_dtype, good_sd_1, good_inner_dict,
                      unzipped_good_sd_1, unzipped_good_inner_dict):
        DASK_ARRAY1, HEADER = \
            unzip_to_dense_dict(good_sd_1, good_header, good_dtype, chunks=(1,3))
        assert DASK_ARRAY1.shape == shape_(good_sd_1)
        assert np.array_equiv(DASK_ARRAY1.compute(), unzipped_good_sd_1)
        assert np.array_equiv(HEADER, np.array(list(good_header)).ravel())
        assert DASK_ARRAY1.dtype == good_dtype
        assert DASK_ARRAY1.chunks == ((1, 1), (3,))

        DASK_ARRAY2, HEADER = \
            unzip_to_dense_dict(good_inner_dict, good_header, good_dtype, chunks=(2,))
        assert DASK_ARRAY2.shape == shape_(good_inner_dict)
        assert np.array_equiv(DASK_ARRAY2.compute(), unzipped_good_inner_dict)
        assert np.array_equiv(HEADER, np.array(list(good_header)).ravel())
        assert DASK_ARRAY2.dtype == good_dtype
        assert DASK_ARRAY2.chunks == ((2,1),)


@pytest.mark.skip
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


    @pytest.mark.parametrize('junk_chunk',
        ([1,2], {1,2}, [1,], 1, 'junk', {'a':1}, lambda x: x, 0, True, np.pi)
    )
    def test_rejects_junk_chunks(self, good_sd_1, junk_chunk):
        with pytest.raises(TypeError):
            unzip_to_dataframe(good_sd_1, chunks=junk_chunk)

    @pytest.mark.parametrize('bad_chunk',
                             ((), (1,2,3), (1,2,3,4), (0, ), (0, 0))
    )
    def test_rejects_bad_chunks(self, good_sd_1, bad_chunk):
        with pytest.raises(ValueError):
            unzip_to_dataframe(good_sd_1, chunks=bad_chunk)

    @pytest.mark.parametrize('empty_sd', ({}, {0:{}}))
    def test_empty_handling(self, empty_sd):
        with pytest.raises(ValueError):
            unzip_to_dataframe(empty_sd)


    @pytest.mark.parametrize('good_header',
        (['a', 'b', 'c'], [['a', 'b', 'c']], np.array(['a', 'b', 'c']),
         {'a', 'b', 'c'}, ('a', 'b', 'c'))
    )
    @pytest.mark.parametrize('good_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    def test_accuracy(self, good_header, good_dtype, good_sd_1, good_inner_dict,
                      unzipped_good_sd_1, unzipped_good_inner_dict):
        DASK_ARRAY1, HEADER = \
            unzip_to_dataframe(good_sd_1, good_header, good_dtype, chunks=(1,3))
        assert DASK_ARRAY1.shape == shape_(good_sd_1)
        assert np.array_equiv(DASK_ARRAY1.compute(), unzipped_good_sd_1)
        assert np.array_equiv(HEADER, np.array(list(good_header)).ravel())
        assert DASK_ARRAY1.dtype == good_dtype
        assert DASK_ARRAY1.chunks == ((1, 1), (3,))

        DASK_ARRAY2, HEADER = \
            unzip_to_dataframe(good_inner_dict, good_header, good_dtype, chunks=(2,))
        assert DASK_ARRAY2.shape == shape_(good_inner_dict)
        assert np.array_equiv(DASK_ARRAY2.compute(), unzipped_good_inner_dict)
        assert np.array_equiv(HEADER, np.array(list(good_header)).ravel())
        assert DASK_ARRAY2.dtype == good_dtype
        assert DASK_ARRAY2.chunks == ((2,1),)


@pytest.mark.skip
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


    @pytest.mark.parametrize('junk_chunk',
        ([1,2], {1,2}, [1,], 1, 'junk', {'a':1}, lambda x: x, 0, True, np.pi)
    )
    def test_rejects_junk_chunks(self, good_sd_1, junk_chunk):
        with pytest.raises(TypeError):
            unzip_to_dask_dataframe(good_sd_1, chunks=junk_chunk)

    @pytest.mark.parametrize('bad_chunk',
                             ((), (1,2,3), (1,2,3,4), (0, ), (0, 0))
    )
    def test_rejects_bad_chunks(self, good_sd_1, bad_chunk):
        with pytest.raises(ValueError):
            unzip_to_dask_dataframe(good_sd_1, chunks=bad_chunk)

    @pytest.mark.parametrize('empty_sd', ({}, {0:{}}))
    def test_empty_handling(self, empty_sd):
        with pytest.raises(ValueError):
            unzip_to_dask_dataframe(empty_sd)


    @pytest.mark.parametrize('good_header',
        (['a', 'b', 'c'], [['a', 'b', 'c']], np.array(['a', 'b', 'c']),
         {'a', 'b', 'c'}, ('a', 'b', 'c'))
    )
    @pytest.mark.parametrize('good_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    def test_accuracy(self, good_header, good_dtype, good_sd_1, good_inner_dict,
                      unzipped_good_sd_1, unzipped_good_inner_dict):
        DASK_ARRAY1, HEADER = \
            unzip_to_dask_dataframe(good_sd_1, good_header, good_dtype, chunks=(1,3))
        assert DASK_ARRAY1.shape == shape_(good_sd_1)
        assert np.array_equiv(DASK_ARRAY1.compute(), unzipped_good_sd_1)
        assert np.array_equiv(HEADER, np.array(list(good_header)).ravel())
        assert DASK_ARRAY1.dtype == good_dtype
        assert DASK_ARRAY1.chunks == ((1, 1), (3,))

        DASK_ARRAY2, HEADER = \
            unzip_to_dask_dataframe(good_inner_dict, good_header, good_dtype, chunks=(2,))
        assert DASK_ARRAY2.shape == shape_(good_inner_dict)
        assert np.array_equiv(DASK_ARRAY2.compute(), unzipped_good_inner_dict)
        assert np.array_equiv(HEADER, np.array(list(good_header)).ravel())
        assert DASK_ARRAY2.dtype == good_dtype
        assert DASK_ARRAY2.chunks == ((2,1),)

