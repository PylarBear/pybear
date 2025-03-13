# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




import pytest
from sparse_dict._transform import (
                                    zip_array,
                                    zip_dask_array,
                                    zip_datadict,
                                    zip_dataframe,
                                    zip_dask_dataframe,
)

from sparse_dict._utils import shape_, outer_len, inner_len

import numpy as np
import pandas as pd
from dask import dataframe as ddf
from dask import array as da



class TestZipArray:

    @pytest.mark.parametrize('junk',
        ('abc', 0, False, None, np.nan, min, lambda x: x, {'a':1})
    )
    def test_rejects_non_list(self, junk):
        with pytest.raises(TypeError):
            zip_array(junk, dtype=float)


    def test_rejects_dask_array(self):
        with pytest.raises(TypeError):
            zip_array(da.array([1,2,3]), dtype=float)


    def test_rejects_dataframes(self):
        with pytest.raises(TypeError):
            zip_array(pd.DataFrame({'a':[1,2], 'b':[3,4]}), dtype=float)

        with pytest.raises(TypeError):
            zip_array(ddf.from_array(np.random.randint(0,10,(3,3))), dtype=float)


    def test_rejects_series(self):
        with pytest.raises(TypeError):
            zip_array(pd.Series([1,2]), dtype=float)

        with pytest.raises(TypeError):
            zip_array(ddf.from_array(np.array([1,2,3])), dtype=float)


    def test_rejects_sparse_dicts(self):
        with pytest.raises(TypeError):
            zip_array({0:{0:1,1:2}, 1:{0:2,1:3}}, dtype=float)


    @pytest.mark.parametrize('array_like',
        ([1,2,3], (1,2,3), {1,2,3}, np.random.randint(1,10,(3,)))
    )
    def test_accepts_1D_array_like(self, array_like):
        zip_array(array_like, dtype=float)


    @pytest.mark.parametrize('array_like',
        ([[1,2],[3,4]], np.random.randint(1,10,(3,3)))
    )
    def test_accepts_2D_array_like(self, array_like):
        zip_array(array_like, dtype=float)


    def test_rejects_3D_np(self):
        with pytest.raises(ValueError):
            zip_array(np.random.randint(1,10,(3,3,3)), dtype=float)


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

            zip_array(ARRAY, dtype=float)


    @pytest.mark.parametrize('_dtype',
        (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
         np.uint64, np.float16, np.float32, np.float64, int, float)
    )
    def test_correct_dtypes(self, _dtype):
        OUTPUT = zip_array(np.random.randint(1,10,(2,2)), dtype=_dtype)

        # outer keys
        assert type(list(OUTPUT.keys())[0]) is int

        # inner keys
        assert type(list(OUTPUT[0].keys())[0]) is int

        # values
        if _dtype == int:
            assert type(OUTPUT[0][0]) is int
        elif _dtype == float:
            assert type(OUTPUT[0][0]) is float
        else:
            assert OUTPUT[0][0].dtype == _dtype


    def test_accepts_ragged_array(self):
        OUTPUT = zip_array([[1,2],[4],[6,7,8]], dtype=float)
        assert OUTPUT == {0:{0:1,1:2}, 1:{0:4}, 2:{0:6,1:7,2:8}}


    def test_accuracy_zeros(self):
        OUTPUT = zip_array([0,0,0], dtype=int)
        assert OUTPUT == {2:0}


    def test_accuracy_non_zeros(self):
        OUTPUT = zip_array([1,1,1], dtype=int)
        assert OUTPUT == {0:1, 1:1, 2:1}


    def test_accuracy_inner(self):
        assert zip_array([1,0,8], dtype=int) == {0:1, 2:8}


    def test_accuracy_outer(self):
        assert zip_array([[1,2], [0,4]], dtype=int) == {0:{0:1,1:2}, 1:{1:4}}


    def test_accuracy_empty(self):
        assert zip_array([], dtype=int) == {}
        assert zip_array([[]], dtype=int) == {0:{}}
        assert zip_array(np.array([]), dtype=int) == {}
        assert zip_array(np.array([[]]), dtype=int) == {0:{}}
        assert zip_array(np.random.randint(0, 10, (0, 0)), dtype=np.int8) == {}
        assert zip_array(np.random.randint(0, 10, (1, 0)), dtype=np.int8) == {0:{}}
        assert zip_array(np.random.randint(0, 10, (0, 1)), dtype=np.int8) == {}


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

        # outer keys
        assert type(list(OUTPUT.keys())[0]) is int

        # inner keys
        assert type(list(OUTPUT[0].keys())[0]) is int

        # values
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
        NP_RESULT = zip_array(NP)
        DA_RESULT = zip_dask_array(DA)
        assert DA_RESULT == NP_RESULT


    def test_handles_a_really_big_dask_array(self):

        DA = da.random.randint(0,1,(10_000,10_000), chunks=(1_000,10_000))
        SD = zip_dask_array(DA)

        assert DA.shape == shape_(SD)


    def test_accuracy_empty(self):
        assert zip_dask_array(da.array([]), dtype=int) == {}
        assert zip_dask_array(da.array([[]]), dtype=int) == {0:{}}
        assert zip_dask_array(da.random.randint(0, 10, (0, 0)), dtype=np.int8) == {}
        assert zip_dask_array(da.random.randint(0, 10, (1, 0)), dtype=np.int8) == {0:{}}
        assert zip_dask_array(da.random.randint(0, 10, (0, 1)), dtype=np.int8) == {}


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

        # outer keys
        assert type(list(SD.keys())[0]) is int

        # inner keys
        assert type(list(SD[0].keys())[0]) is int

        # values
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


    def test_accuracy_empty(self):
        assert zip_datadict({}, dtype=int) == ({}, [[]])
        assert zip_datadict({'a':[]}, dtype=int) == ({0:{}}, [['a']])


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

        # outer keys
        assert type(list(SD.keys())[0]) is int

        # inner keys
        assert type(list(SD[0].keys())[0]) is int

        # values
        if _dtype == int:
            assert type(SD[0][0]) is int
        elif _dtype == float:
            assert type(SD[0][0]) is float
        else:
            assert SD[0][0].dtype == _dtype

        assert np.array_equiv(HEADER, ['A', 'B'])


    def test_accuracy_zeros(self):
        SD, HEADER = zip_dataframe(pd.DataFrame({'A': [0,0,0]}), dtype=int)
        assert SD == {0:{0:0}, 1: {0:0}, 2:{0:0}}
        assert np.array_equiv(HEADER, ['A'])


    def test_accuracy_non_zeros(self):
        SD, HEADER = zip_dataframe(pd.DataFrame({'A':[1,1,1]}), dtype=int)
        assert SD == {0: {0:1}, 1:{0:1}, 2:{0:1}}
        assert np.array_equiv(HEADER, ['A'])


    def test_accuracy_empty(self):
        assert zip_dataframe(pd.DataFrame({}), dtype=int) == ({}, [[]])


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
                                         columns=['A', 'B'],
                                         dtype=_dtype
                            ),
                            chunksize=1
            ),
            dtype=_dtype
        )

        # outer keys
        assert type(list(SD.keys())[0]) is int

        # inner keys
        assert type(list(SD[0].keys())[0]) is int

        # values
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
                                                pd.DataFrame({'A': [0,0,0]}),
                                                npartitions=1,
                                ),
                                dtype=int
        )
        assert SD == {0:{0:0}, 1:{0:0}, 2:{0:0}}
        assert np.array_equiv(HEADER, ['A'])


    def test_accuracy_non_zeros(self):
        SD, HEADER = zip_dask_dataframe(
                    ddf.from_pandas(
                                    pd.DataFrame({'A': [1,1,1]}, dtype=int),
                                    npartitions=1
                    ),
                    dtype=int
        )
        assert SD == {0: {0:1}, 1:{0:1}, 2:{0:1}}
        assert np.array_equiv(HEADER, ['A'])


    def test_accuracy_empty(self):
        empty_dask_df = ddf.from_pandas(
                                        pd.DataFrame({}, dtype=int),
                                        chunksize=1
        )

        assert len(empty_dask_df) == 0
        assert zip_dask_dataframe(empty_dask_df, dtype=int) == ({}, [[]])












