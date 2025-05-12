# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np
import dask.array as da
import dask.dataframe as ddf

from dask_ml.wrappers import Incremental, ParallelPostFit

from pybear.preprocessing._SlimPolyFeatures.SlimPolyFeatures import \
    SlimPolyFeatures as SlimPoly



# TEST DASK Incremental + ParallelPostFit == ONE BIG fit_transform()
class TestDaskIncrementalParallelPostFit:


    @pytest.mark.parametrize('x_format', ['da_array', 'ddf'])
    @pytest.mark.parametrize('y_format', ['da_vector', None])
    @pytest.mark.parametrize('row_chunk', (10, 20))
    @pytest.mark.parametrize('wrappings', ('incr', 'ppf', 'both', 'none'))
    def test_fit_and_transform_accuracy(
        self, wrappings, X_np, y_np, _columns, x_format, y_format, _kwargs,
        _shape, row_chunk
    ):

        # faster without client

        if wrappings == 'incr':
            _test_cls = Incremental(SlimPoly(**_kwargs))
        elif wrappings == 'ppf':
            _test_cls = ParallelPostFit(SlimPoly(**_kwargs))
        elif wrappings == 'both':
            _test_cls = ParallelPostFit(Incremental(SlimPoly(**_kwargs)))
        elif wrappings == 'none':
            _test_cls = SlimPoly(**_kwargs)

        _X_chunks = (row_chunk, _shape[1])
        _X = da.from_array(X_np).rechunk(_X_chunks)

        if x_format == 'da_array':
            pass
        elif x_format == 'ddf':
            _X = ddf.from_dask_array(_X, columns=_columns)
        else:
            raise Exception

        # confirm there is an X
        _X.shape


        if y_format is None:
            _y = None
        elif y_format == 'da_vector':
            _y = da.from_array(y_np, chunks=(row_chunk,))
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
            with pytest.raises(Exception):
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
            assert isinstance(TRFM_X, ddf.DataFrame)
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

        REF_X = RefTestCls.fit_transform(X_np, _y)

        assert isinstance(TRFM_X, np.ndarray)
        assert isinstance(REF_X, np.ndarray)
        assert np.array_equal(TRFM_X, REF_X), \
            f"wrapped output != unwrapped output"




