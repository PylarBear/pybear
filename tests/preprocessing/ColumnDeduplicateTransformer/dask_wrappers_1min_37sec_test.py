# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.preprocessing import ColumnDeduplicateTransformer as CDT

import numpy as np

import pandas as pd
import scipy.sparse as ss

from dask_ml.wrappers import Incremental, ParallelPostFit



bypass = False


# v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
# FIXTURES

@pytest.fixture(scope='session')
def _shape():
    return (20, 10)


@pytest.fixture(scope='module')
def _kwargs():
    return {
        'keep': 'first',
        'do_not_drop': None,
        'conflict': 'raise',
        'rtol': 1e-5,
        'atol': 1e-8,
        'equal_nan': False,
        'n_jobs': -1
    }


@pytest.fixture(scope='module')
def _dum_X(_X_factory, _shape):
    return _X_factory(_dupl=None, _has_nan=False, _dtype='flt', _shape=_shape)


# pizza is this even used
@pytest.fixture(scope='module')
def _std_dupl(_shape):
    return [[0, 4], [3, 5, _shape[1] - 1]]


@pytest.fixture(scope='module')
def _columns(_master_columns, _shape):
    return _master_columns.copy()[:_shape[1]]


@pytest.fixture(scope='module')
def _bad_columns(_master_columns, _shape):
    return _master_columns.copy()[-_shape[1]:]


@pytest.fixture(scope='module')
def _X_pd(_dum_X, _columns):
    return pd.DataFrame(
        data=_dum_X,
        columns=_columns
    )


# END fixtures
# v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^



# TEST DASK Incremental + ParallelPostFit == ONE BIG sklearn fit_transform()
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestDaskIncrementalParallelPostFit:

    # pizza, come back and reread the whole thing, see if it needs anything


    @staticmethod
    @pytest.fixture
    def CDT_not_wrapped(_kwargs):
        return CDT(**_kwargs)

    @staticmethod
    @pytest.fixture
    def CDT_wrapped_parallel(_kwargs):
        return ParallelPostFit(CDT(**_kwargs))

    @staticmethod
    @pytest.fixture
    def CDT_wrapped_incremental(_kwargs):
        return Incremental(CDT(**_kwargs))

    @staticmethod
    @pytest.fixture
    def CDT_wrapped_both(_kwargs):
        return ParallelPostFit(Incremental(CDT(**_kwargs)))


    FORMATS = ['np', 'pddf']
    @pytest.mark.parametrize('x_format', FORMATS + ['csr', 'csc', 'coo'])
    @pytest.mark.parametrize('y_format', FORMATS + ['pdseries', None])
    @pytest.mark.parametrize('wrappings', ('incr', 'ppf', 'both', 'none'))
    def test_fit_and_transform_accuracy(self, wrappings,
        CDT_wrapped_parallel, CDT_wrapped_incremental, CDT_not_wrapped,
        CDT_wrapped_both, _dum_X, _columns, x_format, y_format, _kwargs,
        _shape
    ):

        # no difference with or without Client --- pizza, check this

        # USE NUMERICAL COLUMNS ONLY 24_03_27_11_45_00  --- pizza
        # NotImplementedError: Cannot use auto rechunking with object dtype.
        # We are unable to estimate the size in bytes of object data

        if wrappings == 'incr':
            _test_cls = CDT_wrapped_parallel
        elif wrappings == 'ppf':
            _test_cls = CDT_wrapped_incremental
        elif wrappings == 'both':
            _test_cls = CDT_wrapped_both
        elif wrappings == 'none':
            _test_cls = CDT_not_wrapped

        _X = _dum_X.copy()
        _np_X = _dum_X.copy()
        _chunks = (_shape[0]//5, _shape[1])
        if x_format == 'np':
            pass
        elif x_format == 'pddf':
            _X = pd.DataFrame(data=_X, columns=_columns)
        elif x_format == 'csc':
            _X = ss.csc_array(_X)
        elif x_format == 'csr':
            _X = ss.csr_array(_X)
        elif x_format == 'coo':
            _X = ss.coo_array(_X)
        else:
            raise Exception



        # confirm there is an X
        _X.shape


        y = np.random.randint(0,2,(_shape[0], 2))

        _y = y.copy()
        _np_y = _y.copy()
        if y_format == 'np':
            pass
        elif y_format in ['pddf', 'pdseries']:
            _y = pd.DataFrame(data=_y, columns=['y1', 'y2'])
            if y_format == 'pdseries':
                _y = _y.iloc[:, 0].squeeze()
                assert isinstance(_y, pd.core.series.Series)
                _np_y = _y.to_frame().to_numpy()
        if y_format is None:
            _y = None

        # confirm there is a y
        if _y is not None:
            _y.shape

        _was_fitted = False
        # incr covers fit() so should accept all objects for fits

        # vvv fit vvv

        _test_cls.partial_fit(_X, _y)
        _test_cls.fit(_X, _y)
        _was_fitted = True

        # ^^^ END fit ^^^

        # vvv transform vvv
        if _was_fitted:

            _x_was_transformed = False

            # always TypeError when try to pass y to transform
            with pytest.raises(TypeError):
                _test_cls.transform(_X, _y)

            # always transforms with just X
            TRFM_X = _test_cls.transform(_X)
            _x_was_transformed = True


            if _x_was_transformed:
                if x_format == 'np':
                    assert isinstance(TRFM_X, np.ndarray)
                elif x_format == 'pddf':
                    assert isinstance(TRFM_X, pd.core.frame.DataFrame)
                elif x_format == 'csc':
                    assert isinstance(TRFM_X, ss.csc_array)
                elif x_format == 'csr':
                    assert isinstance(TRFM_X, ss.csr_array)
                elif x_format == 'coo':
                    assert isinstance(TRFM_X, ss.coo_array)
                else:
                    raise Exception


                # CONVERT TO NP ARRAY FOR COMPARISON AGAINST ONE-SHOT fit_trfm()
                try:
                    TRFM_X = TRFM_X.to_numpy()
                except:
                    pass


                try:
                    TRFM_X = TRFM_X.toarray()
                except:
                    pass

                # END CONVERT TO NP ARRAY FOR COMPARISON AGAINST ONE-SHOT fit_trfm()

                FitTransformTestCls = CDT(**_kwargs)

                FT_TRFM_X = FitTransformTestCls.fit_transform(_np_X)

                assert isinstance(TRFM_X, np.ndarray)
                assert isinstance(FT_TRFM_X, np.ndarray)
                assert np.array_equiv(
                        TRFM_X.astype(str), FT_TRFM_X.astype(str)), \
                    (f"transformed X  != transformed np X on single fit/transform")


# END TEST DASK Incremental + ParallelPostFit == ONE BIG sklearn fit_transform()











