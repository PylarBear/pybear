# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from copy import deepcopy

import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as ss

from pybear.preprocessing import InterceptManager as IM

from pybear.utilities import nan_mask, nan_mask_numerical, nan_mask_string



bypass = False


# v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
# FIXTURES

@pytest.fixture(scope='module')
def _constants(_shape):
    return {0: 0, _shape[1]-2: np.nan, _shape[1]-1: 1}


@pytest.fixture(scope='module')
def _X_np(_X_factory, _constants, _shape):
    return _X_factory(
        _has_nan=False,
        _dtype='flt',
        _constants=_constants,
        _shape=_shape
    )

# END fixtures
# v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^


# test input validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestInitValidation:


    # keep ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('junk_keep', (True, False, None, [1,2], {1,2}))
    def test_junk_keep(self, _X_np, _kwargs, junk_keep):

        _kwargs['keep'] = junk_keep

        with pytest.raises(ValueError):
            IM(**_kwargs).fit_transform(_X_np)


    @pytest.mark.parametrize('bad_keep',
        (-1, np.pi, 'rubbish', {1:'trash'}, lambda x: 'junk', min)
    )
    def test_bad_keep(self, _X_np, _kwargs, bad_keep):

        _kwargs['keep'] = bad_keep

        with pytest.raises(ValueError):
            IM(**_kwargs).fit_transform(_X_np)


    @pytest.mark.parametrize('good_keep',
        ('first', 'last', 'random', 'none', 0, {'Intercept': 1},
         lambda x: 0, 'string')
    )
    def test_good_keep(self, _columns, _kwargs, good_keep, _X_np):

        if good_keep == 'string':
            good_keep = _columns[0]

        _kwargs['keep'] = good_keep

        IM(**_kwargs).fit_transform(pd.DataFrame(_X_np, columns=_columns))
    # END keep ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # rtol & atol ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('_param', ('rtol', 'atol'))
    @pytest.mark.parametrize('_junk',
        (None, 'trash', [1,2], {1,2}, {'a':1}, lambda x: x, min)
    )
    def test_junk_rtol_atol(self, _X_np, _kwargs, _param, _junk):

        _kwargs[_param] = _junk

        # non-num are handled by np.allclose, let it raise
        # whatever it will raise
        with pytest.raises(Exception):
            IM(**_kwargs).fit_transform(_X_np)


    @pytest.mark.parametrize('_param', ('rtol', 'atol'))
    @pytest.mark.parametrize('_bad', [-np.pi, -2, -1, True, False])
    def test_bad_rtol_atol(self, _X_np, _kwargs, _param, _bad):

        _kwargs[_param] = _bad

        with pytest.raises(ValueError):
            IM(**_kwargs).fit_transform(_X_np)


    @pytest.mark.parametrize('_param', ('rtol', 'atol'))
    @pytest.mark.parametrize('_good', (1e-5, 1e-6, 1e-1))
    def test_good_rtol_atol(self, _X_np, _kwargs, _param, _good):

        _kwargs[_param] = _good

        IM(**_kwargs).fit_transform(_X_np)

    # END rtol & atol ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # equal_nan ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @pytest.mark.parametrize('_junk',
        (-1, 0, 1, np.pi, None, 'trash', [1, 2], {1, 2}, {'a': 1}, lambda x: x)
    )
    def test_non_bool_equal_nan(self, _X_np, _kwargs, _junk):

        _kwargs['equal_nan'] = _junk

        with pytest.raises(TypeError):
            IM(**_kwargs).fit_transform(_X_np)


    @pytest.mark.parametrize('_equal_nan', [True, False])
    def test_equal_nan_accepts_bool(self, _X_np, _kwargs, _equal_nan):

        _kwargs['equal_nan'] = _equal_nan

        IM(**_kwargs).fit_transform(_X_np)

    # END equal_nan ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # n_jobs ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('junk_n_jobs',
        (True, False, 'trash', [1, 2], {1, 2}, {'a': 1}, lambda x: x, min)
    )
    def test_junk_n_jobs(self, _X_np, _kwargs, junk_n_jobs):

        _kwargs['n_jobs'] = junk_n_jobs

        with pytest.raises(TypeError):
            IM(**_kwargs).fit_transform(_X_np)


    @pytest.mark.parametrize('bad_n_jobs', [-2, 0])
    def test_bad_n_jobs(self, _X_np, _kwargs, bad_n_jobs):

        _kwargs['n_jobs'] = bad_n_jobs

        with pytest.raises(ValueError):
            IM(**_kwargs).fit_transform(_X_np)


    @pytest.mark.parametrize('good_n_jobs', [-1, 1, 10, None])
    def test_good_n_jobs(self, _X_np, _kwargs, good_n_jobs):

        _kwargs['n_jobs'] = good_n_jobs

        IM(**_kwargs).fit_transform(_X_np)

    # END n_jobs ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

# END test input validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


# TEST X ###############################################################

@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestX:

    # - accepts ndarray, pd.DataFrame, pl.DataFrame and all ss
    # - cannot be None
    # - must be 2D
    # - must have at least 1 column
    # - allows nan

    # CONTAINERS #######################################################
    @pytest.mark.parametrize('_junk_X',
        (-1, 0, 1, 3.14, None, 'junk', [0, 1], (1,), {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_X(self, _kwargs, _X_np, _junk_X):

        TestCls = IM(**_kwargs)

        with pytest.raises(ValueError):
            TestCls.partial_fit(_junk_X)

        with pytest.raises(ValueError):
            TestCls.fit(_junk_X)

        with pytest.raises(ValueError):
            TestCls.fit_transform(_junk_X)

        TestCls.fit(_X_np)

        with pytest.raises(ValueError):
            TestCls.transform(_junk_X)

        with pytest.raises(ValueError):
            TestCls.inverse_transform(_junk_X)


    @pytest.mark.parametrize('_format', ('py_list', 'py_tuple'))
    def test_rejects_invalid_container(self, _X_np, _columns, _kwargs, _format):

        TestCls = IM(**_kwargs)

        if _format == 'py_list':
            _X_wip = list(map(list, _X_np.copy()))
        elif _format == 'py_tuple':
            _X_wip = tuple(map(tuple, _X_np.copy()))
        else:
            raise Exception

        with pytest.raises(ValueError):
            TestCls.partial_fit(_X_wip)

        with pytest.raises(ValueError):
            TestCls.fit(_X_wip)

        with pytest.raises(ValueError):
            TestCls.fit_transform(_X_wip)

        TestCls.fit(_X_np)

        with pytest.raises(ValueError):
            TestCls.transform(_X_wip)

        with pytest.raises(ValueError):
            TestCls.inverse_transform(_X_wip)


    @pytest.mark.parametrize('_format',
        (
         'np', 'pd', 'pl', 'csr_matrix', 'csc_matrix', 'coo_matrix', 'dia_matrix',
         'lil_matrix', 'dok_matrix', 'bsr_matrix', 'csr_array', 'csc_array',
         'coo_array', 'dia_array', 'lil_array', 'dok_array', 'bsr_array'
        )
    )
    def test_good_X_container(
        self, _X_factory, _columns, _shape, _kwargs, _constants, _format
    ):
        _X_wip = _X_factory(
            _dupl=None,
            _has_nan=False,
            _format=_format,
            _dtype='flt',
            _columns=_columns,
            _constants=_constants,
            _noise=0,
            _zeros=None,
            _shape=_shape
        )

        _IM = IM(**_kwargs)

        _IM.partial_fit(_X_wip)

        _IM.fit(_X_wip)

        _IM.fit_transform(_X_wip)

        TRFM_X = _IM.transform(_X_wip)

        _IM.inverse_transform(TRFM_X)

    # END CONTAINERS ###################################################


    # SHAPE ############################################################
    @pytest.mark.parametrize('_format', ('np', 'pd', 'pl'))
    def test_rejects_1D(self, _X_np, _kwargs, _format):

        _IM = IM(**_kwargs)

        if _format == 'np':
            _X_wip = _X_np[:, 0]
        elif _format == 'pd':
            _X_wip = pd.Series(_X_np[:, 0])
        elif _format == 'pl':
            _X_wip = pl.Series(_X_np[:, 0])
        else:
            raise Exception

        with pytest.raises(ValueError):
            _IM.partial_fit(_X_wip)

        with pytest.raises(ValueError):
            _IM.fit(_X_wip)

        with pytest.raises(ValueError):
            _IM.fit_transform(_X_wip)

        _IM.fit(_X_np)

        with pytest.raises(ValueError):
            _IM.transform(_X_wip)

        with pytest.raises(ValueError):
            _IM.inverse_transform(_X_wip)


    @pytest.mark.parametrize('_num_cols', (0, 1, 2))
    @pytest.mark.parametrize('_format', ('np', 'pd', 'pl', 'coo_array'))
    def test_X_2D_num_columns(
        self, _X_np, _shape, _kwargs, _columns, _format, _num_cols
    ):

        if _format == 'np':
            _X_wip = _X_np[:, :_num_cols]
        elif _format == 'pd':
            _X_wip = pd.DataFrame(
                _X_np[:, :_num_cols], columns=_columns[:_num_cols]
            )
        elif _format == 'pl':
            _X_wip = pl.DataFrame(
                _X_np[:, :_num_cols], schema=list(_columns[:_num_cols])
            )
        elif _format == 'coo_array':
            _X_wip = ss.csc_array(_X_np[:, :_num_cols])
        else:
            raise Exception

        assert len(_X_wip.shape) == 2
        assert _X_wip.shape[1] == _num_cols

        _IM = IM(**_kwargs)

        if _num_cols == 0:
            with pytest.raises(ValueError):
                _IM.partial_fit(_X_wip)
            with pytest.raises(ValueError):
                _IM.fit(_X_wip)
            with pytest.raises(ValueError):
                _IM.transform(_X_wip)
            with pytest.raises(ValueError):
                _IM.fit_transform(_X_wip)
            with pytest.raises(ValueError):
                _IM.inverse_transform(_X_wip)
        else:
            _IM.partial_fit(_X_wip)
            _IM.fit(_X_wip)
            _IM.transform(_X_wip)
            _IM.fit_transform(_X_wip)
            _IM.inverse_transform(_X_wip)


    @pytest.mark.parametrize('_format', ('np', 'pd', 'pl', 'lil_matrix'))
    def test_rejects_no_samples(self, _shape, _kwargs, _format):

        _IM = IM(**_kwargs)

        _X_base = np.empty((0, _shape[1]), dtype=np.float64)

        if _format == 'np':
            _X_wip = _X_base.copy()
        elif _format == 'pd':
            _X_wip = pd.DataFrame(_X_base)
        elif _format == 'pl':
            _X_wip = pl.DataFrame(_X_base)
        elif _format == 'lil_matrix':
            _X_wip = ss.lil_matrix(_X_base)
        else:
            raise Exception

        # this is caught by if _X.shape[0] == 0 in _val_X

        with pytest.raises(ValueError):
            _IM.partial_fit(_X_wip)

        with pytest.raises(ValueError):
            _IM.fit(_X_wip)

        with pytest.raises(ValueError):
            _IM.fit_transform(_X_wip)

        with pytest.raises(ValueError):
            _IM.transform(_X_wip)

        with pytest.raises(ValueError):
            _IM.inverse_transform(_X_wip)


    @pytest.mark.parametrize('_format', ('np', 'pd', 'pl', 'csr_matrix'))
    @pytest.mark.parametrize('_diff', ('more', 'less', 'same'))
    def test_rejects_bad_num_features(
        self, _X_factory, _shape, _constants, _kwargs, _columns, _X_np,
        _format, _diff
    ):
        # ** ** ** **
        # THERE CANNOT BE "BAD NUM FEATURES" FOR fit & fit_transform
        # THE MECHANISM FOR partial_fit & transform IS DIFFERENT FROM inverse_transform
        # ** ** ** **

        _new_shape_dict = {
            'same': _shape,
            'less': (_shape[0], _shape[1] - 1),
            'more': (_shape[0], 2 * _shape[1])
        }
        _columns_dict = {
            'same': _columns,
            'less': _columns.copy()[:-1],
            'more': np.hstack((_columns, np.char.upper(_columns)))
        }
        _new_constants_dict = {
            'same': _constants,
            'less': {0: 0, _shape[1] - 2: np.nan},
            'more': _constants
        }

        _X_wip = _X_factory(
            _dupl=None,
            _has_nan=False,
            _format=_format,
            _dtype='flt',
            _columns=_columns_dict[_diff],
            _constants=_new_constants_dict[_diff],
            _zeros=0,
            _shape=_new_shape_dict[_diff]
        )

        _IM = IM(**_kwargs)
        _IM.fit(_X_np)

        if _diff == 'same':
            _IM.partial_fit(_X_wip)
            _IM.transform(_X_wip)
        else:
            with pytest.raises(ValueError):
                _IM.partial_fit(_X_wip)
            with pytest.raises(ValueError):
                _IM.transform(_X_wip)

    # END SHAPE ########################################################


    # TEST ValueError WHEN SEES A DF HEADER DIFFERENT FROM FIRST-SEEN HEADER
    @pytest.mark.parametrize('_format', ('pd', 'pl'))
    @pytest.mark.parametrize('fst_fit_columns', ('DF1', 'DF2', 'NO_HDR_DF'))
    @pytest.mark.parametrize('scd_fit_columns', ('DF1', 'DF2', 'NO_HDR_DF'))
    @pytest.mark.parametrize('trfm_columns', ('DF1', 'DF2', 'NO_HDR_DF'))
    def test_except_or_warn_on_different_headers(
        self, _X_factory, _kwargs, _columns, _shape, _format,
        fst_fit_columns, scd_fit_columns, trfm_columns
    ):

        _factory_kwargs = {
            '_dupl':None, '_format':_format, '_dtype':'flt',
            '_has_nan':False, '_constants': None, '_shape':_shape
        }

        # np.flip(_columns) is bad columns
        _col_dict = {'DF1': _columns, 'DF2': np.flip(_columns), 'NO_HDR_DF': None}

        fst_fit_X = _X_factory(_columns=_col_dict[fst_fit_columns], **_factory_kwargs)
        scd_fit_X = _X_factory(_columns=_col_dict[scd_fit_columns], **_factory_kwargs)
        trfm_X = _X_factory(_columns=_col_dict[trfm_columns], **_factory_kwargs)

        TestCls = IM(**_kwargs)

        _objs = [fst_fit_columns, scd_fit_columns, trfm_columns]
        # EXCEPT IF 2 DIFFERENT HEADERS ARE SEEN
        pybear_exception = 0
        pybear_exception += bool('DF1' in _objs and 'DF2' in _objs)
        # POLARS ALWAYS HAS A HEADER
        if _format == 'pl':
            pybear_exception += (len(np.unique(_objs)) > 1)
        # IF FIRST FIT WAS WITH PD NO HEADER, THEN ANYTHING GETS THRU ON
        # SUBSEQUENT partial_fits AND transform
        if _format == 'pd':
            pybear_exception -= bool(fst_fit_columns == 'NO_HDR_DF')
        pybear_exception = max(0, pybear_exception)

        # WARN IF HAS-HEADER AND PD NOT-HEADER BOTH PASSED DURING fits/transform
        # POLARS SHOULDNT GET IN HERE, WILL ALWAYS EXCEPT, ALWAYS HAS A HEADER
        pybear_warn = 0
        if not pybear_exception:
            pybear_warn += ('NO_HDR_DF' in _objs)
            # IF NONE OF THEM HAD A HEADER, THEN NO WARNING
            pybear_warn -= ('DF1' not in _objs and 'DF2' not in _objs)
            pybear_warn = max(0, pybear_warn)

        del _objs

        if pybear_exception:
            # this raises in _check_feature_names
            with pytest.raises(ValueError):
                TestCls.partial_fit(fst_fit_X)
                TestCls.partial_fit(scd_fit_X)
                TestCls.transform(trfm_X)
        elif pybear_warn:
            with pytest.warns():
                TestCls.partial_fit(fst_fit_X)
                TestCls.partial_fit(scd_fit_X)
                TestCls.transform(trfm_X)
        else:
            # SHOULD NOT EXCEPT OR WARN
            TestCls.partial_fit(fst_fit_X)
            TestCls.partial_fit(scd_fit_X)
            TestCls.transform(trfm_X)

    # END TEST ValueError WHEN SEES A DF HEADER DIFFERENT FROM FIRST-SEEN HEADER

# END TEST X ###########################################################


@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestPartialFit:

    #     def partial_fit(
    #         self,
    #         X: DataContainer,
    #         y: any=None
    #     ) -> Self:


    @pytest.mark.parametrize('_format',
        (
         'np', 'pd', 'pl', 'csr_matrix', 'csc_matrix', 'coo_matrix', 'dia_matrix',
         'lil_matrix', 'dok_matrix', 'bsr_matrix', 'csr_array', 'csc_array',
         'coo_array', 'dia_array', 'lil_array', 'dok_array', 'bsr_array'
        )
    )
    def test_X_is_not_mutated(
        self, _X_factory, _columns, _shape, _kwargs, _constants, _format
    ):
        _X_wip = _X_factory(
            _dupl=None,
            _has_nan=False,
            _format=_format,
            _dtype='flt',
            _columns=_columns,
            _constants=_constants,
            _noise=0,
            _zeros=None,
            _shape=_shape
        )

        if _format == 'np':
            assert _X_wip.flags['C_CONTIGUOUS'] is True

        try:
            _X_wip_before = _X_wip.copy()
        except:
            _X_wip_before = _X_wip.clone()


        # verify _X_wip does not mutate in partial_fit()
        _IM = IM(**_kwargs)
        _IM.partial_fit(_X_wip)


        # ASSERTIONS v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v
        assert isinstance(_X_wip, type(_X_wip_before))
        assert _X_wip.shape == _X_wip_before.shape

        if isinstance(_X_wip, np.ndarray):
            assert _X_wip.flags['C_CONTIGUOUS'] is True
            assert np.array_equal(_X_wip_before, _X_wip, equal_nan=True)
        elif hasattr(_X_wip, 'columns'):  # DATAFRAMES
            assert _X_wip.equals(_X_wip_before)
        elif hasattr(_X_wip_before, 'toarray'):
            assert np.array_equal(
                _X_wip.toarray(), _X_wip_before.toarray(), equal_nan=True
            )
        else:
            raise Exception


    @pytest.mark.parametrize('_stuff',
        (
            -1,0,1, np.pi, True, False, None, 'trash', [1,2], {1,2}, {'a':1},
            lambda x: x, min
        )
    )
    def test_fit_partial_fit_accept_Y_equals_anything(self, _kwargs, _X_np, _stuff):
        IM(**_kwargs).fit(_X_np, _stuff)
        IM(**_kwargs).partial_fit(_X_np, _stuff)


    def test_conditional_access_to_partial_fit_and_fit(self, _X_np, _kwargs):

        TestCls = IM(**_kwargs)

        # 1) partial_fit() should allow unlimited number of subsequent partial_fits()
        for _ in range(5):
            TestCls.partial_fit(_X_np)

        TestCls._reset()

        # 2) one call to fit() should allow subsequent attempts to partial_fit()
        TestCls.fit(_X_np)
        TestCls.partial_fit(_X_np)

        TestCls._reset()

        # 3) one call to fit() should allow later attempts to fit() (2nd fit will reset)
        TestCls.fit(_X_np)
        TestCls.fit(_X_np)

        TestCls._reset()

        # 4) a call to fit() after a previous partial_fit() should be allowed
        TestCls.partial_fit(_X_np)
        TestCls.fit(_X_np)

        TestCls._reset()

        # 5) fit_transform() should allow calls ad libido
        for _ in range(5):
            TestCls.fit_transform(_X_np)

        del TestCls


    @pytest.mark.parametrize('_keep',
        ('first', 'last', 'random', 'none', 0, lambda x: 0, {'Intercept':1})
    )
    @pytest.mark.parametrize('_equal_nan', (True, False))
    def test_many_partial_fits_equal_one_big_fit(
        self, _X_np, _kwargs, _shape, _keep, _equal_nan
    ):

        # **** **** **** **** **** **** **** **** **** **** **** **** ****
        # THIS TEST IS CRITICAL FOR VERIFYING THAT transform PULLS THE
        # SAME COLUMN INDICES FOR ALL CALLS TO transform() WHEN
        # keep=='random'
        # **** **** **** **** **** **** **** **** **** **** **** **** ****

        _kwargs['keep'] = _keep
        _kwargs['equal_nan'] = _equal_nan

        # ** ** ** ** ** ** ** ** ** ** **
        # TEST THAT ONE-SHOT partial_fit/transform == ONE-SHOT fit/transform
        OneShotPartialFitTestCls = IM(**_kwargs).partial_fit(_X_np)

        OneShotFullFitTestCls = IM(**_kwargs).fit(_X_np)

        _ = OneShotPartialFitTestCls.constant_columns_
        __ = OneShotFullFitTestCls.constant_columns_
        assert np.array_equal(list(_.keys()), list(__.keys()))
        # need to turn to strings because of nans
        # _X_np _has_nan=False, but constants have a column of np.nans
        assert np.array_equal(
            list(map(str, _.values())), list(map(str, __.values()))
        )
        del _, __

        ONE_SHOT_PARTIAL_FIT_TRFM_X = \
            OneShotPartialFitTestCls.transform(_X_np, copy=True)

        ONE_SHOT_FULL_FIT_TRFM_X = \
            OneShotFullFitTestCls.transform(_X_np, copy=True)

        assert ONE_SHOT_PARTIAL_FIT_TRFM_X.shape == \
               ONE_SHOT_FULL_FIT_TRFM_X.shape

        if _keep != 'random':
            # this has np.nan in it, convert to str
            assert np.array_equal(
                ONE_SHOT_PARTIAL_FIT_TRFM_X.astype(str),
                ONE_SHOT_FULL_FIT_TRFM_X.astype(str)
            ), f"one shot partial fit trfm X != one shot full fit trfm X"


        del ONE_SHOT_PARTIAL_FIT_TRFM_X, ONE_SHOT_FULL_FIT_TRFM_X

        # END TEST THAT ONE-SHOT partial_fit/transform==ONE-SHOT fit/transform
        # ** ** ** ** ** ** ** ** ** ** **

        # ** ** ** ** ** ** ** ** ** ** **
        # TEST PARTIAL FIT CONSTANTS ARE THE SAME WHEN FULL DATA IS partial_fit() 2X
        SingleFitTestClass = IM(**_kwargs).fit(_X_np)
        _ = SingleFitTestClass.constant_columns_

        DoublePartialFitTestClass = IM(**_kwargs)
        DoublePartialFitTestClass.partial_fit(_X_np)
        __ = DoublePartialFitTestClass.constant_columns_
        DoublePartialFitTestClass.partial_fit(_X_np)
        ___ = DoublePartialFitTestClass.constant_columns_

        assert np.array_equal(list(_.keys()), list(__.keys()))
        assert np.array_equal(list(_.keys()), list(___.keys()))
        assert np.array_equal(
            list(map(str, _.values())),
            list(map(str, __.values()))
        )
        assert np.array_equal(
            list(map(str, _.values())),
            list(map(str, ___.values()))
        )

        del _, __, ___, SingleFitTestClass, DoublePartialFitTestClass

        # END PARTIAL FIT CONSTANTS ARE THE SAME WHEN FULL DATA IS partial_fit() 2X
        # ** ** ** ** ** ** ** ** ** ** **

        # ** ** ** ** ** ** ** ** ** ** **# ** ** ** ** ** ** ** ** ** ** **
        # ** ** ** ** ** ** ** ** ** ** **# ** ** ** ** ** ** ** ** ** ** **
        # TEST MANY PARTIAL FITS == ONE BIG FIT

        # STORE CHUNKS TO ENSURE THEY STACK BACK TO THE ORIGINAL X
        _chunks = 5
        X_CHUNK_HOLDER = []
        for row_chunk in range(_chunks):
            _mask_start = row_chunk * _shape[0] // _chunks
            _mask_end = (row_chunk + 1) * _shape[0] // _chunks
            X_CHUNK_HOLDER.append(_X_np[_mask_start:_mask_end, :])
        del _mask_start, _mask_end

        assert np.array_equiv(
            np.vstack(X_CHUNK_HOLDER).astype(str), _X_np.astype(str)
        ), \
            f"agglomerated X chunks != original X"

        PartialFitTestCls = IM(**_kwargs)
        OneShotFitTransformTestCls = IM(**_kwargs)

        # PIECEMEAL PARTIAL FIT ******************************************
        for X_CHUNK in X_CHUNK_HOLDER:
            PartialFitTestCls.partial_fit(X_CHUNK)

        # PIECEMEAL TRANSFORM ******************************************
        # THIS CANT BE UNDER THE partial_fit LOOP, ALL FITS MUST BE DONE
        # BEFORE DOING ANY TRFMS
        PARTIAL_TRFM_X_HOLDER = []
        for X_CHUNK in X_CHUNK_HOLDER:
            PARTIAL_TRFM_X_HOLDER.append(PartialFitTestCls.transform(X_CHUNK))

        # AGGLOMERATE PARTIAL TRFMS FROM PARTIAL FIT
        FULL_TRFM_X_FROM_PARTIAL_FIT_PARTIAL_TRFM = \
            np.vstack(PARTIAL_TRFM_X_HOLDER)

        del PARTIAL_TRFM_X_HOLDER
        # END PIECEMEAL TRANSFORM **************************************

        # DO ONE-SHOT TRANSFORM OF X ON THE PARTIALLY FIT INSTANCE
        FULL_TRFM_X_FROM_PARTIAL_FIT_ONESHOT_TRFM = \
            PartialFitTestCls.transform(_X_np)

        del PartialFitTestCls


        if _keep != 'random':

            # ONE-SHOT FIT TRANSFORM
            FULL_TRFM_X_ONE_SHOT_FIT_TRANSFORM = \
                OneShotFitTransformTestCls.fit_transform(_X_np)

            del OneShotFitTransformTestCls

            # ASSERT ALL AGGLOMERATED X TRFMS ARE EQUAL
            assert np.array_equiv(
                    FULL_TRFM_X_ONE_SHOT_FIT_TRANSFORM.astype(str),
                    FULL_TRFM_X_FROM_PARTIAL_FIT_PARTIAL_TRFM.astype(str)
                ), \
                (f"compiled trfm X from partial fit / partial trfm != "
                 f"one-shot fit/trfm X")

            assert np.array_equiv(
                FULL_TRFM_X_ONE_SHOT_FIT_TRANSFORM.astype(str),
                FULL_TRFM_X_FROM_PARTIAL_FIT_ONESHOT_TRFM.astype(str)
                ), (f"trfm X from partial fits / one-shot trfm != one-shot "
                    f"fit/trfm X")

        elif _keep == 'random':

            assert np.array_equiv(
                    FULL_TRFM_X_FROM_PARTIAL_FIT_PARTIAL_TRFM.astype(str),
                    FULL_TRFM_X_FROM_PARTIAL_FIT_ONESHOT_TRFM.astype(str)
                ), (f"trfm X from partial fit / partial trfm != "
                 f"trfm X from partial fit / one-shot trfm/")


    @pytest.mark.parametrize('_format', ('np', ))
    @pytest.mark.parametrize('_dtype', ('int', 'flt', 'int', 'obj', 'hybrid'))
    @pytest.mark.parametrize('_has_nan', (False, 5))
    def test_constant_columns_accuracy_over_many_partial_fits(
        self, _kwargs, _X_factory, _format, _dtype, _has_nan
    ):

        # verify correct progression of reported constants as partial fits are done.
        # rig a set of arrays that have progressively decreasing constants

        # skip impossible conditions -- -- -- -- -- -- -- -- -- -- -- --
        if _format not in ('np', 'pd', 'pl') and _dtype not in ('flt', 'int'):
            pytest.skip(reason=f"cant put non-num in scipy sparse")
        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        _chunk_shape = (50, 20)

        _start_constants = {3: 1, 5: 1, _chunk_shape[1] - 2: 1}

        _new_kwargs = deepcopy(_kwargs)
        _new_kwargs['equal_nan'] = True

        PartialFitTestCls = IM(**_new_kwargs)

        # build a pool of non-constants to fill the constants in X along the way
        # build a starting data object for first partial fit, using full constants
        # build a y vector
        # do a verification partial_fit, assert reported constants for original X
        # make a holder for all the different _wip_Xs, to do one big fit at the end
        # for however many times u want to do this:
        #   randomly replace one of the constants with non-constant column
        #   partial_fit
        #   assert reported constants - should be one less (the randomly chosen)
        # at the very end, stack all the _wip_Xs, do one big fit, verify constants

        _pool_X = _X_factory(
            _dupl=None,
            _has_nan=_has_nan,
            _format=_format,
            _dtype=_dtype,
            _columns=None,
            _constants=None,   # <============
            _noise=1e-9,
            _zeros=None,
            _shape=_chunk_shape
        )

        _wip_X = _X_factory(
            _dupl=None,
            _has_nan=_has_nan,
            _format=_format,
            _dtype=_dtype,
            _columns=None,
            _constants=_start_constants,   # <===========
            _noise=0,
            _zeros=None,
            _shape=_chunk_shape
        )

        y_np = np.random.randint(0, 2, (_chunk_shape[0]))

        # verify IM sees the constant columns correctly ** * ** * ** *
        # this also sets the original constants columns in PartialFitTestCls
        _constant_columns = \
            PartialFitTestCls.partial_fit(_wip_X, y_np).constant_columns_
        assert len(_constant_columns) == len(_start_constants)
        for idx, v in _start_constants.items():
            if str(v) == 'nan':
                assert str(v) == str(_constant_columns[idx])
            else:
                assert v == _constant_columns[idx]
        del _constant_columns
        # END verify IM sees the constant columns correctly ** * ** * **

        # create a holder for the the original constant column idxs
        _const_pool = list(_start_constants)

        X_HOLDER = []
        X_HOLDER.append(_wip_X)

        # take out only half of the constants (arbitrary) v^v^v^v^v^v^v^v^v^v^v
        for trial in range(len(_const_pool)//2):

            random_const_idx = np.random.choice(_const_pool, 1, replace=False)[0]

            # take the random constant of out _start_constants and _const_pool,
            # and take a column out of the X pool to patch the constant in _wip_X
            _start_constants.pop(random_const_idx)
            _const_pool.remove(random_const_idx)

            # column from X should be constant, column from pool shouldnt be
            # but verify anyway ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
            _from_X = _wip_X[:, random_const_idx]
            _from_pool = _pool_X[:, random_const_idx]
            assert not np.array_equal(
                _from_X[np.logical_not(nan_mask(_from_X))],
                _from_pool[np.logical_not(nan_mask(_from_pool))]
            )
            del _from_X, _from_pool
            # END verify ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

            _wip_X[:, random_const_idx] = _pool_X[:, random_const_idx].copy()

            X_HOLDER.append(_wip_X)

            # fit PartialFitTestCls on the new _wip_X
            # verify correctly reported constants after this partial_fit
            _constant_columns = \
                PartialFitTestCls.partial_fit(_wip_X, y_np).constant_columns_
            assert len(_constant_columns) == len(_start_constants)
            for idx, v in _start_constants.items():
                if str(v) == 'nan':
                    assert str(v) == str(_constant_columns[idx])
                else:
                    assert v == _constant_columns[idx]
            del _constant_columns

        # END take out only half of the constants (arbitrary) v^v^v^v^v^v^v^v^v

        # we now have full X_HOLDER, which holds _wip_Xs with progressively
        # fewer columns of constants
        # and PartialFitTestCls, which was fit sequentially on the _wip_Xs


        _partial_fit_constant_columns = PartialFitTestCls.constant_columns_
        # do a one-shot fit, compare results
        # stack all the _wip_Xs
        OneShotFitTestCls = IM(**_new_kwargs).fit(np.vstack(X_HOLDER), y_np)
        _one_shot_constant_columns = OneShotFitTestCls.constant_columns_
        # remember that _start_constants has constant idxs popped out of it
        # as non-constant columns were put into _wip_X
        assert len(_one_shot_constant_columns) == len(_start_constants)
        assert len(_partial_fit_constant_columns) == len(_start_constants)
        for idx, v in _start_constants.items():
            if str(v) == 'nan':
                assert str(v) == str(_one_shot_constant_columns[idx])
                assert str(v) == str(_partial_fit_constant_columns[idx])
            else:
                assert v == _one_shot_constant_columns[idx]
                assert v == _partial_fit_constant_columns[idx]


    # dont really need to test accuracy, see _partial_fit


@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestTransform:

    #     def transform(
    #         self,
    #         X: DataContainer,
    #         *,
    #         copy: bool = None
    #     ) -> DataContainer:

    # - only accepts ndarray, pd.DataFrame, and all ss
    # - cannot be None
    # - must be 2D
    # - must have at least 2 columns
    # - must have at least 1 sample
    # - num columns must equal num columns seen during fit
    # - allows nan
    # - output is C contiguous
    # - validates all instance attrs -- this isnt tested here, see _validation


    @pytest.mark.parametrize('_copy',
        (-1, 0, 1, 3.14, True, False, None, 'junk', [0, 1], (1,), {'a': 1}, min)
    )
    def test_copy_validation(self, _X_np, _shape, _kwargs, _copy):

        _IM = IM(**_kwargs)
        _IM.fit(_X_np)

        if isinstance(_copy, (bool, type(None))):
            _IM.transform(_X_np, copy=_copy)
        else:
            with pytest.raises(TypeError):
                _IM.transform(_X_np, copy=_copy)


    @pytest.mark.parametrize('_format', ('np', 'pd', 'pl', 'csc_array'))
    @pytest.mark.parametrize('output_type', (None, 'default', 'pandas', 'polars'))
    def test_output_types(
        self, _X_factory, _columns, _shape, _kwargs, _constants, _format,
        output_type
    ):
        _X_wip = _X_factory(
            _dupl=None, _has_nan=False, _format=_format, _dtype='flt',
            _columns=_columns if _format in ['pd', 'pl'] else None,
            _constants=_constants, _noise=0, _zeros=None, _shape=_shape
        )

        TestCls = IM(**_kwargs)
        TestCls.set_output(transform=output_type)
        TRFM_X = TestCls.fit(_X_wip)

        TRFM_X = TestCls.transform(_X_wip)

        # if output_type is None, should return same type as given
        # if output_type is 'default', should return np array no matter what given
        # if output_type is 'pandas' or 'polars', should return pd df no matter what given
        _output_type_dict = {
            None: type(_X_wip), 'default': np.ndarray, 'polars': pl.DataFrame,
            'pandas': pd.core.frame.DataFrame
        }
        assert isinstance(TRFM_X, _output_type_dict[output_type]), \
            (f"output_type is {output_type}, X output type ({type(TRFM_X)}) != "
             f"X input type ({type(_X_wip)})")

    # TEST OUTPUT TYPES ####################################################


    @pytest.mark.parametrize('_format',
        (
         'np', 'pd', 'pl', 'csr_matrix', 'csc_matrix', 'coo_matrix', 'dia_matrix',
         'lil_matrix', 'dok_matrix', 'bsr_matrix', 'csr_array', 'csc_array',
         'coo_array', 'dia_array', 'lil_array', 'dok_array', 'bsr_array'
        )
    )
    def test_X_is_not_mutated(
            self, _X_factory, _columns, _shape, _kwargs, _constants, _format
    ):

        _X_wip = _X_factory(
            _dupl=None,
            _has_nan=False,
            _format=_format,
            _dtype='flt',
            _columns=_columns,
            _constants=_constants,
            _noise=0,
            _zeros=None,
            _shape=_shape
        )

        if _format == 'np':
            assert _X_wip.flags['C_CONTIGUOUS'] is True

        try:
            _X_wip_before = _X_wip.copy()
        except:
            _X_wip_before = _X_wip.clone()


        _IM = IM(**_kwargs)

        # verify _X_wip does not mutate in transform() with copy=True
        _IM.fit(_X_wip)
        TRFM_X = _IM.transform(_X_wip, copy=True)


        # ASSERTIONS v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v
        assert isinstance(_X_wip, type(_X_wip_before))
        assert _X_wip.shape == _X_wip_before.shape

        if isinstance(_X_wip_before, np.ndarray):
            assert np.array_equal(_X_wip_before, _X_wip, equal_nan=True)
        elif hasattr(_X_wip_before, 'columns'):    # DATAFRAMES
            assert _X_wip.equals(_X_wip_before)
        elif hasattr(_X_wip_before, 'toarray'):
            assert np.array_equal(
                _X_wip.toarray(), _X_wip_before.toarray(), equal_nan=True
            )
        else:
            raise Exception


    @pytest.mark.parametrize('_equal_nan', (True, False))
    def test_one_all_nans(self, _X_factory, _kwargs, _shape, _equal_nan):

        _X = _X_factory(
            _dupl=None,
            _has_nan=False,
            _format='np',
            _dtype='flt',
            _columns=None,
            _constants={0:1, 1:2},
            _zeros=None,
            _shape=(_shape[0], 3)    # <====== 3 columns
        )

        # set a column to all nans
        _X[:, -1] = np.nan
        # verify last column is all nans
        assert all(nan_mask_numerical(_X[:, -1]))

        # conftest _kwargs 'keep' == 'first'
        _kwargs['equal_nan'] = _equal_nan
        TRFM_X = IM(**_kwargs).fit_transform(_X)

        if _equal_nan:
            # last 2 columns should drop, should have 1 column, not np.nan
            assert TRFM_X.shape[1] == 1
            assert np.array_equal(TRFM_X[:, 0], _X[:, 0])
            assert not any(nan_mask_numerical(TRFM_X[:, -1]))
        elif not _equal_nan:
            # 2nd column should drop, should have 2 columns, last is all np.nan
            assert TRFM_X.shape[1] == 2
            assert np.array_equal(TRFM_X[:, 0], _X[:, 0])
            assert all(nan_mask_numerical(TRFM_X[:, -1]))


    @pytest.mark.parametrize('_equal_nan', (True, False))
    def test_two_all_nans(self, _X_factory, _kwargs, _shape, _equal_nan):

        _X = _X_factory(
            _dupl=None,
            _has_nan=False,
            _format='np',
            _dtype='flt',
            _columns=None,
            _constants={0:1, 1:2},
            _zeros=None,
            _shape=(_shape[0], 4)    # <======== 4 columns
        )

        # set last 2 columns to all nans
        _X[:, [-2, -1]] = np.nan
        # verify last columns are all nans
        assert all(nan_mask_numerical(_X[:, -1]))
        assert all(nan_mask_numerical(_X[:, -2]))

        # conftest _kwargs 'keep'=='first'
        _kwargs['equal_nan'] = _equal_nan
        TRFM_X = IM(**_kwargs).fit_transform(_X)

        if _equal_nan:
            # last 3 columns should drop, should have 1 column, not np.nan
            assert TRFM_X.shape[1] == 1
            assert np.array_equal(TRFM_X[:, 0], _X[:, 0])
            assert not any(nan_mask_numerical(TRFM_X[:, -1]))
        elif not _equal_nan:
            # only 2nd columns should drop, should have 3 columns
            assert TRFM_X.shape[1] == 3
            assert np.array_equal(TRFM_X[:, 0], _X[:, 0])
            assert all(nan_mask_numerical(TRFM_X[:, -1]))
            assert all(nan_mask_numerical(TRFM_X[:, -2]))


    @pytest.mark.parametrize('_dtype', ('str', 'obj'))
    def test_fit_transform_floats_as_str_dtypes(
        self, _X_factory, _dtype, _shape, _constants
    ):

        # make an array of floats....
        _wip_X = _X_factory(
            _dupl=None,
            _has_nan=False,
            _format='np',
            _dtype='flt',
            _columns=None,
            _constants=_constants,
            _zeros=0,
            _shape=_shape
        )

        # set dtype
        _wip_X = _wip_X.astype('<U20' if _dtype == 'str' else object)

        _IM = IM(
            keep='last',
            equal_nan=True,
            rtol=1e-5,
            atol=1e-8,
            n_jobs=1
        )

        out = _IM.fit_transform(_wip_X)

        assert isinstance(out, np.ndarray)

        # keep == 'last'!
        _ref_column_mask = np.ones((_shape[1],)).astype(bool)
        MASK = [i in _constants for i in range(_shape[1])]
        _ref_column_mask[MASK] = False
        _ref_column_mask[sorted(list(_constants))[-1]] = True
        del MASK

        assert np.array_equal(_IM.column_mask_, _ref_column_mask)


    # '_same' also tests when scipy sparse is all zeros

    @pytest.mark.parametrize('x_format',
        ('np', 'pd', 'pl', 'coo_array', 'csc_array', 'csr_array')
    )
    @pytest.mark.parametrize('keep',
        ('first', 'last', 'random', 'none', 'int', 'string', 'callable', 'dict'))
    @pytest.mark.parametrize('same_or_diff', ('_same', '_diff'))
    def test_all_columns_the_same_or_different(
        self, _X_factory, _kwargs, _columns, _shape, _constants, same_or_diff,
        keep, x_format
    ):

        # skip impossible conditions -- -- -- -- -- -- -- -- -- -- -- --
        if keep == 'string' and x_format not in ['pd', 'pl']:
                pytest.skip(reason=f"cant use str keep when not df")
        # END skip impossible conditions -- -- -- -- -- -- -- -- -- -- --

        # set init params ** * ** * ** * ** * ** * ** * ** * ** * ** *
        _wip_constants = deepcopy(_constants)   # _constants is module scope!

        if _kwargs['equal_nan'] is False:
            _wip_constants = {
                k: v for k, v in _wip_constants.items() if str(v) != 'nan'
            }

        keep = {
            'string': _columns[0], 'dict': {'Bias': np.e},
            'callable': lambda x: list(_wip_constants.keys())[0],
            'int': list(_wip_constants.keys())[0]
        }.get(keep, keep)   # if keep not in dict, keep does not change

        # this must be after 'keep' management!
        _kwargs['keep'] = keep
        # END set init params ** * ** * ** * ** * ** * ** * ** * ** * **

        TestCls = IM(**_kwargs)

        # BUILD X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
        # set the constant coluns for X_factory
        # if '_same', make all the columns the same constant
        _value = _wip_constants[list(_wip_constants.keys())[0]]
        if same_or_diff == '_same':
            _dupl = [list(range(_shape[1]))]
            _wip_constants = {i: _value for i in range(_shape[1])}
        else:
            _dupl = None
            # _wip_constants stays the same

        TEST_X = _X_factory(
            _dupl=_dupl,
            _has_nan=False,
            _format=x_format,
            _dtype='flt',
            _columns=_columns if x_format in ['pd', 'pl'] else None,
            _constants=_wip_constants,
            _noise=0,
            _zeros=None,
            _shape=_shape
        )
        # END BUILD X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # after making X, finalize _wip_constants to use it as referee
        if _kwargs['equal_nan'] is False and str(_value) == 'nan':
            _wip_constants = {}

        if keep == 'none' and same_or_diff == '_same':
            with pytest.raises(ValueError):
                # raises if all columns will be deleted
                TestCls.fit_transform(TEST_X)
            pytest.skip(reason=f"cant do anymore tests without fit")
        else:
            TRFM_X = TestCls.fit_transform(TEST_X)

        # ASSERTIONS v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v
        assert TestCls.constant_columns_ == _wip_constants, \
            f"TestCls.constant_columns_ != _wip_constants"

        if keep != 'none' and not isinstance(keep, dict):
            if same_or_diff == '_same':
                # if all are constant, all but 1 column is deleted
                assert TRFM_X.shape[1] == 1
            elif same_or_diff == '_diff':
                assert TRFM_X.shape[1] == _shape[1] - len(_wip_constants) + 1
        elif isinstance(keep, dict):
            if same_or_diff == '_same':
                # if all are constant, all original are deleted, append new
                assert TRFM_X.shape[1] == 1
            elif same_or_diff == '_diff':
                assert TRFM_X.shape[1] == _shape[1] - len(_wip_constants) + 1
        elif keep == 'none':
            if same_or_diff == '_same':
                raise Exception(f"shouldnt be in here")
                # this was tested above under a pytest.raises. should raise
                # because all columns will be removed.
            elif same_or_diff == '_diff':
                assert TRFM_X.shape[1] == _shape[1] - len(_wip_constants)
        else:
            raise Exception(f'algorithm failure')


    # pizza See IMTransform_accuracy for accuracy tests
    # pizza revisit accuracy test.... is proably redundant with transform_test


@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestFitTransform:

    @pytest.mark.parametrize('_format', ('np', 'pd', 'pl', 'csc_array'))
    @pytest.mark.parametrize('output_type', (None, 'default', 'pandas', 'polars'))
    def test_output_types(
            self, _X_factory, _columns, _shape, _kwargs, _constants, _format,
            output_type
    ):
        _X_wip = _X_factory(
            _dupl=None, _has_nan=False, _format=_format, _dtype='flt',
            _columns=_columns if _format in ['pd', 'pl'] else None,
            _constants=_constants, _noise=0, _zeros=None, _shape=_shape
        )

        TestCls = IM(**_kwargs)
        TestCls.set_output(transform=output_type)

        TRFM_X = TestCls.fit_transform(_X_wip)

        # if output_type is None, should return same type as given
        # if output_type is 'default', should return np array no matter what given
        # if output_type is 'pandas' or 'polars', should return pd df no matter what given
        _output_type_dict = {
            None: type(_X_wip), 'default': np.ndarray, 'polars': pl.DataFrame,
            'pandas': pd.core.frame.DataFrame
        }
        assert isinstance(TRFM_X, _output_type_dict[output_type]), \
            (f"output_type is {output_type}, X output type ({type(TRFM_X)}) != "
             f"X input type ({type(_X_wip)})")


@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestInverseTransform:

    #     def inverse_transform(
    #         self,
    #         X: DataContainer,
    #         *,
    #         copy: bool = None
    #     ) -> DataContainer:

    # - only accepts ndarray, pd.DataFrame, and all ss
    # - cannot be None
    # - must be 2D
    # - must have at least 1 column
    # - must have at least 1 sample
    # - num columns must equal num columns in column_mask_
    # - allows nan
    # - output is C contiguous
    # - no instance attr validation


    @pytest.mark.parametrize('_copy',
        (-1, 0, 1, 3.14, True, False, None, 'junk', [0, 1], (1,), {'a': 1}, min)
    )
    def test_copy_validation(self, _X_np, _shape, _kwargs, _copy):

        _IM = IM(**_kwargs)
        _IM.fit(_X_np)

        if isinstance(_copy, (bool, type(None))):
            _IM.inverse_transform(_X_np[:, _IM.column_mask_], copy=_copy)
        else:
            with pytest.raises(TypeError):
                _IM.inverse_transform(_X_np[:, _IM.column_mask_], copy=_copy)


    @pytest.mark.parametrize('_format',
        (
            'np', 'pd', 'pl', 'csr_matrix', 'csc_matrix', 'coo_matrix', 'dia_matrix',
            'lil_matrix', 'dok_matrix', 'bsr_matrix', 'csr_array', 'csc_array',
            'coo_array', 'dia_array', 'lil_array', 'dok_array', 'bsr_array'
        )
    )
    @pytest.mark.parametrize('_dtype', ('int', 'flt', 'str', 'obj', 'hybrid'))
    @pytest.mark.parametrize('_has_nan', (True, False))
    @pytest.mark.parametrize('_keep',
        (
            'first', 'last', 'random', 'none', 0, 'string', lambda x: 0,
            {'Intercept': None}    # replace None in the function
        )
    )
    @pytest.mark.parametrize('_constants', ('constants1', 'constants2'))
    @pytest.mark.parametrize('_copy', (True, False))
    def test_accuracy(
        self, _X_factory, _columns, _kwargs, _shape, _format, _dtype,
        _has_nan, _keep, _constants, _copy
    ):

        # pizza while fixing the pd.NA except, maybe hack this down
        # this is probably redundant with inverse_transform_test!

        # may not need to test accuracy here, see _inverse_transform,
        # but it is pretty straightforward. affirms the IM class
        # inverse_transform method works correctly, above and beyond just
        # the _inverse_transform function called within.

        # set_output does not control the output container for inverse_transform
        # the output container is always the same as passed

        # skip impossible conditions -- -- -- -- -- -- -- -- -- -- -- -- -- --
        if _format not in ('np', 'pd', 'pl') and _dtype not in ('int', 'flt'):
            pytest.skip(reason=f"scipy sparse cant take non-numeric")
        # END skip impossible conditions -- -- -- -- -- -- -- -- -- -- -- -- --

        if isinstance(_keep, dict):
            _key = list(_keep.keys())[0]
            if _dtype in ('int', 'flt'):
                _keep[_key] = 1
            else:
                _keep[_key] = '1'
            del _key
        elif _keep == 'string':
            if _format == 'pd':
                _keep = _columns[0]
            else:
                pytest.skip(reason=f"can only have str 'keep' with pd df")

        # build X ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        if _constants == 'constants1':
            _constants = {}
        elif _constants == 'constants2':
            if _dtype in ('int', 'flt'):
                _constants = {0: -1, 7:np.nan, 9:1}
            else:
                _constants = {0: 'a', 2: 'b', 9:'nan'}
        else:
            raise Exception

        _X_wip = _X_factory(
            _dupl=None,
            _has_nan=_has_nan,
            _format=_format,
            _dtype=_dtype,
            _columns=_columns,
            _constants=_constants,
            _zeros=0,
            _shape=_shape
        )

        if _format == 'np':
            _base_X = _X_wip.copy()
        elif _format in ['pd', 'pl']:
            _base_X = _X_wip.to_numpy()
        elif hasattr(_X_wip, 'toarray'):
            _base_X = _X_wip.toarray()
        else:
            raise Exception

        # this needs to be here for funky pd nans. verified 25_05_24.
        try:
            _base_X[nan_mask(_base_X)] = np.nan
        except:
            pass
        # END build X ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        try:
            _X_wip_before_inv_tr = _X_wip.copy()
        except:
            _X_wip_before_inv_tr = _X_wip.clone()

        _kwargs['keep'] = _keep
        _kwargs['equal_nan'] = True

        _IM = IM(**_kwargs)

        # fit v v v v v v v v v v v v v v v v v v v v
        _raise_for_keep_not_constant = 0
        _warn_for_keep_not_constant = 0
        # can only take literal keeps when no constants (but warns)
        if isinstance(_keep, int):
            if _keep not in _constants:
                _raise_for_keep_not_constant += 1
        elif callable(_keep):
            if _keep(_X_wip) not in _constants:
                _raise_for_keep_not_constant += 1
        elif isinstance(_keep, str):
            if _keep in ('first', 'last', 'random', 'none'):
                if len(_constants) == 0:
                    _warn_for_keep_not_constant += 1
            else:
                # should only be for pd, should have skipped above for non-pd
                if len(_constants):
                    _idx = np.arange(len(_columns))[_columns==_keep][0]
                    if _idx not in _constants:
                        _raise_for_keep_not_constant += 1
                else:
                    _raise_for_keep_not_constant += 1

        if _warn_for_keep_not_constant:
            with pytest.warns():
                _IM.fit(_X_wip)
        elif _raise_for_keep_not_constant:
            with pytest.raises(ValueError):
                _IM.fit(_X_wip)
            pytest.skip(reason=f"cannot continue test without fit")
        else:
            _IM.fit(_X_wip)

        del _raise_for_keep_not_constant
        # fit ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^

        # transform v v v v v v v v v v v v v v v v v v
        TRFM_X = _IM.transform(_X_wip, copy=True)
        assert isinstance(TRFM_X, type(_X_wip))
        # transform ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^

        # inverse transform v v v v v v v v v v v v v v v
        INV_TRFM_X = _IM.inverse_transform(
            X=TRFM_X,
            copy=_copy
        )
        # inverse transform ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^

        # output container is same as passed
        assert isinstance(INV_TRFM_X, type(_X_wip))

        # if output is numpy, order is C
        if isinstance(INV_TRFM_X, np.ndarray):
            assert INV_TRFM_X.flags['C_CONTIGUOUS'] is True

        # verify dimension of output
        assert INV_TRFM_X.shape[0] == TRFM_X.shape[0], \
            f"rows in output of inverse_transform() do not match input rows"
        assert INV_TRFM_X.shape[1] == _IM.n_features_in_, \
            (f"num features in output of inverse_transform() do not match "
             f"originally fitted columns")

        # convert everything to ndarray to use array_equal
        if isinstance(TRFM_X, np.ndarray):
            NP_TRFM_X = TRFM_X
            NP_INV_TRFM_X = INV_TRFM_X
        elif isinstance(TRFM_X, pd.core.frame.DataFrame):
            NP_TRFM_X = TRFM_X.to_numpy()
            NP_INV_TRFM_X = INV_TRFM_X.to_numpy()
        elif isinstance(TRFM_X, pl.DataFrame):
            # Polars uses zero-copy conversion when possible, meaning the
            # underlying memory is still controlled by Polars and marked
            # as read-only. NumPy and Pandas may inherit this read-only
            # flag, preventing modifications.
            # THE ORDER IS IMPORTANT HERE. CONVERT TO PANDAS FIRST, THEN COPY.
            NP_TRFM_X = TRFM_X.to_pandas().to_numpy()
            NP_INV_TRFM_X = INV_TRFM_X.to_pandas().to_numpy()
        elif hasattr(TRFM_X, 'toarray'):
            NP_TRFM_X = TRFM_X.toarray()
            NP_INV_TRFM_X = INV_TRFM_X.toarray()
        else:
            raise Exception

        assert isinstance(NP_TRFM_X, np.ndarray)
        assert isinstance(NP_INV_TRFM_X, np.ndarray)

        # v v v v assert output is equal to original pre-transform data v v v v

        if isinstance(NP_INV_TRFM_X, (pd.core.frame.DataFrame, pl.DataFrame)):
            assert np.array_equal(NP_INV_TRFM_X.columns, _columns)

        # manage equal_nan for num or str
        try:
            NP_INV_TRFM_X.astype(np.float64)
            is_num = True
        except:
            is_num = False

        # inverse_transform is unable to know where any nans were in the
        # original data, so it puts in columns with no nans. np.array_equal
        # with equal_nan does not work for that situation. need to apply
        # nans to the inverse transform to make the nans the same as the
        # input so that a non-complicated comparison can be made.
        # (otherwise would probably need to do for loops and apply a mask
        # to each individual column)
        try:
            NP_INV_TRFM_X[nan_mask(_base_X)] = np.nan
        except:
            pass

        if is_num:
            # when num

            assert NP_INV_TRFM_X.shape == _base_X.shape, \
                f"{NP_INV_TRFM_X.shape=}, {_base_X.shape=}"

            for _c_idx in range(_base_X.shape[1]):
                assert np.array_equal(
                    NP_INV_TRFM_X[:, _c_idx].astype(np.float64),
                    _base_X[:, _c_idx].astype(np.float64),
                    equal_nan=True
                ), \
                    (f"inverse transform of transformed data does not equal "
                     f"original data")

            if not isinstance(_keep, dict):
                assert np.array_equal(
                    NP_TRFM_X.astype(np.float64),
                    NP_INV_TRFM_X[:, _IM.column_mask_].astype(np.float64),
                    equal_nan=True
                ), \
                    (f"output of inverse_transform() does not reduce back to "
                     f"the input of transform()")
        else:
            # when str
            # IM converts all nan-likes to np.nan, need to standardize them
            # in the inputs here for array_equal against the outputs
            # for string data, array_equal does not accept equal_nan param
            _base_X[nan_mask_string(_base_X)] = 'nan'
            NP_TRFM_X[nan_mask_string(NP_TRFM_X)] = 'nan'
            NP_INV_TRFM_X[nan_mask_string(NP_INV_TRFM_X)] = 'nan'

            assert NP_INV_TRFM_X.shape == _base_X.shape, \
                f"{NP_INV_TRFM_X.shape=}, {_base_X.shape=}"

            assert np.array_equal(NP_INV_TRFM_X, _base_X), \
                f"inverse transform of transformed data != original data"

            if not isinstance(_keep, dict):
                assert np.array_equal(
                    NP_TRFM_X,
                    NP_INV_TRFM_X[:, _IM.column_mask_]
                ), \
                    (f"output of inverse_transform() does not reduce back to "
                     f"the input of transform()")


        # verify _X_wip does not mutate in inverse_transform()
        # save the headache of dealing with array_equal with nans and
        # non-numeric data, just do numeric.
        if _copy is True and _dtype in ('flt', 'int'):

            assert isinstance(_X_wip, type(_X_wip_before_inv_tr))
            assert _X_wip.shape == _X_wip_before_inv_tr.shape

            if isinstance(_X_wip_before_inv_tr, np.ndarray):
                assert np.array_equal(
                    _X_wip_before_inv_tr, _X_wip, equal_nan=True
                )
                assert _X_wip.flags == _X_wip_before_inv_tr.flags
            elif isinstance(
                _X_wip_before_inv_tr,
                (pd.core.frame.DataFrame, pl.DataFrame)
            ):
                assert _X_wip.equals(_X_wip_before_inv_tr)
            elif hasattr(_X_wip_before_inv_tr, 'toarray'):
                assert np.array_equal(
                    _X_wip.toarray(),
                    _X_wip_before_inv_tr.toarray(),
                    equal_nan=True
                )
            else:
                raise Exception


    @pytest.mark.parametrize('_format',
        (
         'np', 'pd', 'pl', 'csr_matrix', 'csc_matrix', 'coo_matrix', 'dia_matrix',
         'lil_matrix', 'dok_matrix', 'bsr_matrix', 'csr_array', 'csc_array',
         'coo_array', 'dia_array', 'lil_array', 'dok_array', 'bsr_array'
        )
    )
    def test_X_is_not_mutated(
        self, _X_factory, _columns, _shape, _kwargs, _constants, _format
    ):
        _X_wip = _X_factory(
            _dupl=None,
            _has_nan=False,
            _format=_format,
            _dtype='flt',
            _columns=_columns,
            _constants=_constants,
            _noise=0,
            _zeros=None,
            _shape=_shape
        )

        if _format == 'np':
            assert _X_wip.flags['C_CONTIGUOUS'] is True


        try:
            _X_wip_before = _X_wip.copy()
        except:
            _X_wip_before = _X_wip.clone()


        # verify _X_wip does not mutate in inverse_transform
        _IM = IM(**_kwargs)
        _IM.fit(_X_wip)
        TRFM_X = _IM.transform(_X_wip, copy=True)
        assert isinstance(TRFM_X, type(_X_wip))
        assert TRFM_X.shape[1] < _X_wip.shape[1]
        INV_TRFM_X = _IM.inverse_transform(TRFM_X)


        # ASSERTIONS v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v
        assert isinstance(_X_wip, type(_X_wip_before))
        assert _X_wip.shape == _X_wip_before.shape

        if hasattr(_X_wip_before, 'toarray'):
            assert np.array_equal(
                _X_wip.toarray(), _X_wip_before.toarray(), equal_nan=True
            )
        elif isinstance(_X_wip_before, pd.core.frame.DataFrame):
            assert _X_wip.equals(_X_wip_before)
        else:
            assert np.array_equal(
                _X_wip_before, _X_wip, equal_nan=True
            )


    @pytest.mark.parametrize('_format', ('np', 'pd', 'pl', 'csr_matrix'))
    @pytest.mark.parametrize('_diff', ('more', 'less', 'same'))
    def test_rejects_bad_num_features(
            self, _X_factory, _shape, _constants, _X_np, _kwargs, _columns,
            _format, _diff
    ):
        # ** ** ** **
        # THERE CANNOT BE "BAD NUM FEATURES" FOR fit & fit_transform
        # THE MECHANISM FOR partial_fit & transform IS DIFFERENT FROM inverse_transform
        # ** ** ** **

        _new_shape_dict = {
            'same': _shape,
            'less': (_shape[0], _shape[1] - 1),
            'more': (_shape[0], 2 * _shape[1])
        }
        _columns_dict = {
            'same': _columns,
            'less': _columns.copy()[:-1],
            'more': np.hstack((_columns, np.char.upper(_columns)))
        }
        _new_constants_dict = {
            'same': _constants,
            'less': {0: 0, _shape[1] - 2: np.nan},
            'more': _constants
        }

        _X_wip = _X_factory(
            _dupl=None,
            _has_nan=False,
            _format=_format,
            _dtype='flt',
            _columns=_columns_dict[_diff],
            _constants=_new_constants_dict[_diff],
            _zeros=0,
            _shape=_new_shape_dict[_diff]
        )

        _IM = IM(**_kwargs)
        _IM.fit(_X_np)

        if _diff == 'same':
            _IM.partial_fit(_X_wip)
            _IM.transform(_X_wip)
        else:
            with pytest.raises(ValueError):
                _IM.partial_fit(_X_wip)
            with pytest.raises(ValueError):
                _IM.transform(_X_wip)

        # FROM INVERSE_TRANSFORM
        _IM.fit(_X_np)

        # build TRFM_X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
        TRFM_X = _IM.transform(_X_np)
        TRFM_MASK = _IM.column_mask_

        if _diff == 'same':
            if _format == 'pd':
                TRFM_X = pd.DataFrame(data=TRFM_X, columns=_columns[TRFM_MASK])
            elif _format == 'pl':
                TRFM_X = pl.from_numpy(data=TRFM_X, schema=list(_columns[TRFM_MASK]))
        elif _diff == 'less':
            TRFM_X = TRFM_X[:, :-1]
            if _format == 'pd':
                TRFM_X = pd.DataFrame(data=TRFM_X, columns=_columns[TRFM_MASK][:-1])
            elif _format == 'pl':
                TRFM_X = pl.from_numpy(data=TRFM_X, schema=list(_columns[TRFM_MASK][:-1]))
        elif _diff == 'more':
            TRFM_X = np.hstack((TRFM_X, TRFM_X))
            if _format == 'pd':
                _COLUMNS = np.hstack((_columns[TRFM_MASK], np.char.upper(_columns[TRFM_MASK])))
                TRFM_X = pd.DataFrame(data=TRFM_X, columns=_COLUMNS)
            elif _format == 'pl':
                _COLUMNS = np.hstack((_columns[TRFM_MASK], np.char.upper(_columns[TRFM_MASK])))
                TRFM_X = pl.from_numpy(data=TRFM_X, schema=list(_COLUMNS))
        else:
            raise Exception
        # END build TRFM_X ** * ** * ** * ** * ** * ** * ** * ** * ** * **

        # Test the inverse_transform operation ** ** ** ** ** ** **
        if _diff == 'same':
            _IM.inverse_transform(TRFM_X)
        else:
            with pytest.raises(ValueError):
                _IM.inverse_transform(TRFM_X)
        # END Test the inverse_transform operation ** ** ** ** ** ** **

        del _IM, TRFM_X, TRFM_MASK
        # END FROM INVERSE_TRANSFORM




