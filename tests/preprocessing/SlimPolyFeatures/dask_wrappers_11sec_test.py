# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.preprocessing.SlimPolyFeatures.SlimPolyFeatures import \
    SlimPolyFeatures as SlimPoly

import numpy as np

from dask_ml.wrappers import Incremental, ParallelPostFit
import dask.array as da
import dask.dataframe as ddf
import dask_expr._collection as ddf2

from distributed import Client





bypass = False


# v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
# FIXTURES

@pytest.fixture(scope='session')
def _shape():
    return (40, 4)


@pytest.fixture(scope='module')
def _kwargs():
    return {
        'degree': 2,
        'min_degree': 1,
        'scan_X': False,
        'keep': 'first',
        'interaction_only': False,
        'sparse_output': False,
        'feature_name_combiner': "as_indices",
        'equal_nan': True,
        'rtol': 1e-5,
        'atol': 1e-8,
        'n_jobs': 1
    }



@pytest.fixture(scope='module')
def _X_np(_X_factory, _shape):
    
    return _X_factory(
        _dupl=None,
        _has_nan=False,
        _dtype='flt',
        _shape=_shape
    )


@pytest.fixture(scope='module')
def _columns(_master_columns, _shape):
    return _master_columns.copy()[:_shape[1]]


# END fixtures
# v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^



# TEST DASK Incremental + ParallelPostFit == ONE BIG fit_transform()
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestDaskIncrementalParallelPostFit:


    @staticmethod
    @pytest.fixture
    def SlimPoly_not_wrapped(_kwargs):
        return SlimPoly(**_kwargs)

    @staticmethod
    @pytest.fixture
    def SlimPoly_wrapped_parallel(_kwargs):
        return ParallelPostFit(SlimPoly(**_kwargs))

    @staticmethod
    @pytest.fixture
    def SlimPoly_wrapped_incremental(_kwargs):
        return Incremental(SlimPoly(**_kwargs))

    @staticmethod
    @pytest.fixture
    def SlimPoly_wrapped_both(_kwargs):
        return ParallelPostFit(Incremental(SlimPoly(**_kwargs)))

    @staticmethod
    @pytest.fixture(scope='module')
    def _client():
        client = Client(n_workers=1, threads_per_worker=1) # 0:42
        yield client
        client.close()


    @pytest.mark.parametrize('x_format', ['da_array', 'ddf'])
    @pytest.mark.parametrize('y_format', ['da_vector', None])
    @pytest.mark.parametrize('row_chunk', (10, 20))
    @pytest.mark.parametrize('wrappings', ('incr', 'ppf', 'both', 'none'))
    def test_fit_and_transform_accuracy(self, wrappings, SlimPoly_wrapped_parallel,
        SlimPoly_wrapped_incremental, SlimPoly_not_wrapped, SlimPoly_wrapped_both,
        _X_np, _columns, x_format, y_format, _kwargs, _shape, row_chunk, #_client
    ):

        # faster without client

        if wrappings == 'incr':
            _test_cls = SlimPoly_wrapped_incremental
        elif wrappings == 'ppf':
            _test_cls = SlimPoly_wrapped_parallel
        elif wrappings == 'both':
            _test_cls = SlimPoly_wrapped_both
        elif wrappings == 'none':
            _test_cls = SlimPoly_not_wrapped

        _X_chunks = (row_chunk, _shape[1])
        _X = da.array(_X_np.copy()).rechunk(_X_chunks)
        _X_np = _X_np.copy()
        if x_format == 'da_array':
            pass
        elif x_format == 'ddf':
            _X = ddf.from_dask_array(_X, columns=_columns)
        else:
            raise Exception

        # confirm there is an X
        _X.shape


        _y_chunks = (row_chunk,)
        if y_format is None:
            _y = None
            _y_np = None
        elif y_format == 'da_vector':
            _y = da.random.randint(0, 2, _shape[0], chunks=_y_chunks)
            _y_np = _y.compute()
        else:
            raise Exception

        # confirm there is a y
        if _y is not None:
            _y.shape

        _was_fitted = False
        # incr covers fit() so should accept all objects for fits

        # vvv fit vvv

        if wrappings in ['none', 'ppf']:
            # without any wrappers, should except for trying to put
            # dask objects into SlimPoly
            with pytest.raises(TypeError):
                _test_cls.partial_fit(_X, _y)
            pytest.skip(reason=f"cannot do more tests if not fit")
        elif wrappings in ['incr', 'both']:
            # dask object being fit by chunks into Incremental wrapper
            _test_cls.partial_fit(_X, _y)
        else:
            raise Exception

        # ^^^ END fit ^^^

        # vvv transform vvv

        # always TypeError when try to pass y to wrapper's transform
        with pytest.raises(TypeError):
            _test_cls.transform(_X, _y)

        # always transforms with just X
        TRFM_X = _test_cls.transform(_X)


        if x_format == 'da_array':
            assert isinstance(TRFM_X, da.core.Array)
        elif x_format == 'ddf':
            assert isinstance(TRFM_X, ddf2.DataFrame)
        else:
            raise Exception

        # ^^^ transform ^^^

        # CONVERT TO NP ARRAY FOR COMPARISON AGAINST REF fit_trfm()
        try:
            TRFM_X = TRFM_X.compute()
        except:
            pass

        try:
            TRFM_X = TRFM_X.to_numpy()
        except:
            pass

        # END CONVERT TO NP ARRAY FOR COMPARISON AGAINST REF fit_trfm()

        RefTestCls = SlimPoly(**_kwargs)

        REF_X = RefTestCls.fit_transform(_X_np, _y_np)

        assert isinstance(TRFM_X, np.ndarray)
        assert isinstance(REF_X, np.ndarray)
        assert np.array_equal(TRFM_X, REF_X), \
            f"wrapped output != unwrapped output"














