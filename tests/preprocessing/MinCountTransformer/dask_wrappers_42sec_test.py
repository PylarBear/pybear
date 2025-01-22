# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

from uuid import uuid4

import numpy as np
import pandas as pd

from pybear.preprocessing.MinCountTransformer.MinCountTransformer import \
    MinCountTransformer as MCT

import dask.array as da
import dask.dataframe as ddf
import dask_expr._collection as ddf2

from dask_ml.wrappers import Incremental, ParallelPostFit


# pizza come back to this, probably should split into multiple tests.
# compare against the other pybear transformers.

# TEST DASK Incremental + ParallelPostFit == ONE BIG fit_transform()
class TestDaskIncrementalParallelPostFit:


    @staticmethod
    @pytest.fixture
    def _shape():
        return (200, 20)

    @staticmethod
    @pytest.fixture
    def MCT_not_wrapped(_args, _kwargs):
        return MCT(*_args, **_kwargs)

    @staticmethod
    @pytest.fixture
    def MCT_wrapped_parallel(_args, _kwargs):
        return ParallelPostFit(MCT(*_args, **_kwargs))

    @staticmethod
    @pytest.fixture
    def MCT_wrapped_incremental(_args, _kwargs):
        return Incremental(MCT(*_args, **_kwargs))

    @staticmethod
    @pytest.fixture
    def MCT_wrapped_both(_args, _kwargs):
        return ParallelPostFit(Incremental(MCT(*_args, **_kwargs)))

    @staticmethod
    @pytest.fixture(scope='function')
    def _args():
        return [5]

    @staticmethod
    @pytest.fixture(scope='function')
    def _kwargs():
        return {
            'ignore_float_columns': False,
            'ignore_non_binary_integer_columns': False,
            'ignore_columns': None,
            'ignore_nan': False,
            'delete_axis_0': False,
            'handle_as_bool': None,
            'reject_unseen_values': True,
            'max_recursions': 1,
            'n_jobs': 4
        }

    @staticmethod
    @pytest.fixture
    def X(_shape):
        return np.random.randint(0, 10, _shape)


    @staticmethod
    @pytest.fixture
    def y(_shape):
        return np.random.randint(0, 2, (_shape[0], 2))


    @staticmethod
    @pytest.fixture
    def COLUMNS(_shape):
        return [str(uuid4())[:5] for _ in range(_shape[1])]


    FORMATS = ['da', 'ddf_df', 'ddf_series', 'np', 'pddf', 'pdseries']

    @pytest.mark.xfail(reason=f"pizza cant explain dask_ml changing arrays")
    @pytest.mark.parametrize('x_format', FORMATS)
    @pytest.mark.parametrize('y_format', FORMATS + [None])
    @pytest.mark.parametrize('wrappings', ('incr', 'ppf', 'both', 'none'))
    def test_always_fits_X_y_always_excepts_transform_with_y(self, wrappings,
        MCT_wrapped_parallel, MCT_wrapped_incremental, MCT_not_wrapped,
        MCT_wrapped_both, X, COLUMNS, y, x_format, y_format, _args, _kwargs
    ):

        # faster without Client, verified 24_10_29

        # USE NUMERICAL COLUMNS ONLY 24_03_27_11_45_00
        # NotImplementedError: Cannot use auto rechunking with object dtype.
        # We are unable to estimate the size in bytes of object data


        if wrappings == 'incr':
            _test_cls = MCT_wrapped_incremental
        elif wrappings == 'ppf':
            _test_cls = MCT_wrapped_parallel
        elif wrappings == 'both':
            _test_cls = MCT_wrapped_both
        elif wrappings == 'none':
            _test_cls = MCT_not_wrapped
        else:
            raise Exception

        _X = X.copy()
        _np_X = _X.copy()
        _chunks = (_X.shape[0]//5, _X.shape[1])
        if x_format in ['pddf', 'pdseries']:
            _X = pd.DataFrame(data=_X, columns=COLUMNS)
        if x_format == 'pdseries':
            _X = _X.iloc[:, 0].squeeze()
            assert isinstance(_X, pd.core.series.Series)
            _np_X = _X.to_frame().to_numpy()
        if x_format in ['da', 'ddf_df', 'ddf_series']:
            _X = da.from_array(_X, chunks=_chunks)
        if x_format in ['ddf_df', 'ddf_series']:
            _X = ddf.from_array(_X, chunksize=_chunks)
        if x_format == 'ddf_series':
            _X = _X.iloc[:, 0].squeeze()
            assert isinstance(_X, (ddf.core.Series, ddf2.Series))
            _np_X = _X.compute().to_frame().to_numpy()

        # confirm there is an X
        _X.shape

        _y = y.copy()
        _np_y = _y.copy()
        _chunks = (_y.shape[0]//5, 2)
        if y_format in ['pddf', 'pdseries']:
            _y = pd.DataFrame(data=_y, columns=['y1', 'y2'])
        if y_format == 'pdseries':
            _y = _y.iloc[:, 0].squeeze()
            assert isinstance(_y, pd.core.series.Series)
            _np_y = _y.to_frame().to_numpy()
        if y_format in ['da', 'ddf_df', 'ddf_series']:
            _y = da.from_array(_y, chunks=_chunks)
        if y_format in ['ddf_df', 'ddf_series']:
            _y = ddf.from_array(_y, chunksize = _chunks)
        if y_format == 'ddf_series':
            _y = _y.iloc[:, 0].squeeze()
            assert isinstance(_y, (ddf.core.Series, ddf2.Series))
            _np_y = _y.compute().to_frame().to_numpy()
        if y_format is None:
            _y = None

        # confirm there is a y
        if _y is not None:
            _y.shape

        _was_fitted = False
        # incr covers fit() so should accept all objects for fits
        _dask = ['da', 'ddf_df', 'ddf_series']
        _non_dask = ['np', 'pddf', 'pdseries']

        a = x_format in _dask and y_format in _non_dask
        b = x_format in _non_dask and y_format in _dask
        if x_format in _non_dask and y_format in _non_dask + [None]:
            # always takes non-dask objects with mixed & matched formats
            _test_cls.partial_fit(_X, _y)
            _test_cls.fit(_X, _y)
        elif wrappings == 'none':
            # unwrapped always raises TypeError on any dask object
            # this error is controlled by pybear, validation of X in partial_fit
            with pytest.raises(TypeError):
                _test_cls.partial_fit(_X, _y)
            with pytest.raises(TypeError):
                _test_cls.fit(_X, _y)
            pytest.skip(reason=f"if unable to fit, cannot proceed with tests")
        elif wrappings in ['incr', 'both']:
            # partial_fit is wrapped
            if (a + b) == 1:  # X & y are mixed dask & non-dask
                # Incremental does not allow mixed dask/non-dask X/y containers
                # this is controlled by dask_ml, let is raise whatever
                with pytest.raises(Exception):
                    _test_cls.partial_fit(_X, _y)
                with pytest.raises(Exception):
                    _test_cls.fit(_X, _y)
                pytest.skip(reason=f"if unable to fit, cannot proceed with tests")
            else:
                # if both dask containers, good to go
                _test_cls.partial_fit(_X, _y)
                _test_cls.fit(_X, _y)

        elif wrappings == 'ppf':
            # when partial_fit is not wrapped, partial_fit cannot take dask objects
            # this error is controlled by pybear, validation of X in partial_fit
            with pytest.raises(TypeError):
                _test_cls.partial_fit(_X, _y)
            with pytest.raises(TypeError):
                _test_cls.fit(_X, _y)
            pytest.skip(reason=f"if unable to fit, cannot proceed with tests")
        else:
            raise Exception

        del _dask, _non_dask

        # ^^^ END fit ^^^

        # vvv transform vvv

        if x_format not in ['pdseries', 'ddf_series']:
            assert _X.shape[1] == X.shape[1]

        # always TypeError when try to pass y with ParallelPostFit
        _y_was_transformed = False
        if wrappings in ['ppf', 'both', 'incr']:

            with pytest.raises(TypeError):
                _test_cls.transform(_X, _y)

            # always transforms with just X
            # pizza this is failing because dask_ml wrappers somehow started
            # changing a 200x20 array to 1x20.
            # print(f'bearpizza before transform print {_X.shape=}')
            # print(f'bearpizza before transform print {type(_X)=}')
            TRFM_X = _test_cls.transform(_X)

        elif wrappings in ['none']:

            if _y is not None:
                _test_cls.transform(_X, _y)
                TRFM_X, TRFM_Y = _test_cls.fit_transform(_X, _y)
                _y_was_transformed = True
            else:
                _test_cls.transform(_X)
                TRFM_X = _test_cls.fit_transform(_X, _y)


        if x_format == 'np':
            assert isinstance(TRFM_X, np.ndarray)
        if x_format == 'pddf':
            assert isinstance(TRFM_X, pd.core.frame.DataFrame)
        if x_format == 'pdseries':
            assert isinstance(TRFM_X, pd.core.series.Series)
        if x_format == 'da' and wrappings == 'none':
            assert isinstance(TRFM_X, np.ndarray)
        elif x_format == 'da':
            assert isinstance(TRFM_X, da.core.Array)
        if x_format == 'ddf_df' and wrappings == 'none':
            assert isinstance(TRFM_X, pd.core.frame.DataFrame)
        elif x_format == 'ddf_df':
            assert isinstance(TRFM_X, (ddf.core.DataFrame, ddf2.DataFrame))
        if x_format == 'ddf_series' and wrappings == 'none':
            assert isinstance(TRFM_X, pd.core.series.Series)
        elif x_format == 'ddf_series':
            assert isinstance(TRFM_X, (ddf.core.Series, ddf2.Series))

        if _y_was_transformed:
            if y_format == 'np':
                assert isinstance(TRFM_Y, np.ndarray)
            if y_format == 'pddf':
                assert isinstance(TRFM_Y, pd.core.frame.DataFrame)
            if y_format == 'pdseries':
                assert isinstance(TRFM_Y, pd.core.series.Series)
            if y_format == 'da' and wrappings == 'none':
                assert isinstance(TRFM_Y, np.ndarray)
            elif y_format == 'da':
                assert isinstance(TRFM_Y, da.core.Array)
            if y_format == 'ddf_df' and wrappings == 'none':
                assert isinstance(TRFM_Y, pd.core.frame.DataFrame)
            elif y_format == 'ddf_df':
                assert isinstance(TRFM_Y,
                    (ddf.core.DataFrame, ddf2.DataFrame)
                )
            if y_format == 'ddf_series' and wrappings == 'none':
                assert isinstance(TRFM_Y, pd.core.series.Series)
            elif y_format == 'ddf_series':
                assert isinstance(TRFM_Y, (ddf.core.Series, ddf2.Series))

        # CONVERT TO NP ARRAY FOR COMPARISON AGAINST ONE-SHOT fit_trfm()
        try:
            TRFM_X = TRFM_X.to_frame()
        except:
            pass

        try:
            TRFM_X = TRFM_X.compute()
        except:
            pass

        try:
            TRFM_X = TRFM_X.to_numpy()
        except:
            pass

        if _y_was_transformed:

            try:
                TRFM_Y = TRFM_Y.to_frame()
            except:
                pass

            try:
                TRFM_Y = TRFM_Y.compute()
            except:
                pass

            try:
                TRFM_Y = TRFM_Y.to_numpy()
            except:
                pass

        # END CONVERT TO NP ARRAY FOR COMPARISON AGAINST ONE-SHOT fit_trfm()

        FitTransformTestCls = MCT(*_args, **_kwargs)
        if _y_was_transformed:
            FT_TRFM_X, FT_TRFM_Y = \
                FitTransformTestCls.fit_transform(_np_X, _np_y)
        else:
            FT_TRFM_X = FitTransformTestCls.fit_transform(_np_X)

        assert isinstance(TRFM_X, np.ndarray)
        assert isinstance(FT_TRFM_X, np.ndarray)
        assert np.array_equiv(
                TRFM_X.astype(str), FT_TRFM_X.astype(str)), \
            (f"transformed X  != transformed np X on single fit/transform")

        if _y_was_transformed:
            assert isinstance(TRFM_Y, np.ndarray)
            assert isinstance(FT_TRFM_Y, np.ndarray)
            assert np.array_equiv(
                TRFM_Y.astype(str),
                FT_TRFM_Y.astype(str)
            ), f"transformed Y != transformed np Y on single fit/transform"

# END TEST DASK Incremental + ParallelPostFit == ONE BIG fit_transform()







































