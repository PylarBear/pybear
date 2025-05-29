# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import uuid

import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as ss

from pybear.preprocessing._SlimPolyFeatures.SlimPolyFeatures import \
    SlimPolyFeatures as SlimPoly

from pybear.utilities import nan_mask_numerical, nan_mask




bypass = False


# test input validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestInitValidation:


    # degree ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('junk_degree',
        (None, [1,2], {1,2}, (1,2), {'a':1}, lambda x: x)
    )
    def test_junk_degree(self, X_np, _kwargs, junk_degree):

        _kwargs['degree'] = junk_degree

        with pytest.raises(ValueError):
            SlimPoly(**_kwargs).fit_transform(X_np)


    @pytest.mark.parametrize('bad_degree',
        (-1, 1, np.pi, True, False)
    )
    def test_bad_degree(self, X_np, _kwargs, bad_degree):

        # degree lower bound of 2 is hard coded, so 1 is bad

        _kwargs['degree'] = bad_degree

        with pytest.raises(ValueError):
            SlimPoly(**_kwargs).fit_transform(X_np)


    @pytest.mark.parametrize('good_degree', (2,3))
    def test_good_degree(self, X_pd, _columns, _kwargs, good_degree):

        _kwargs['degree'] = good_degree

        SlimPoly(**_kwargs).fit_transform(X_pd)
    # END degree ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # min_degree ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    @pytest.mark.parametrize('junk_min_degree',
        (None, [1,2], {1,2}, (1,2), {'a':1}, lambda x: x)
    )
    def test_junk_min_degree(self, X_np, _kwargs, junk_min_degree):

        _kwargs['min_degree'] = junk_min_degree

        with pytest.raises(ValueError):
            SlimPoly(**_kwargs).fit_transform(X_np)


    @pytest.mark.parametrize('bad_min_degree',
        (-1, 0, np.pi, True, False)
    )
    def test_bad_min_degree(self, X_np, _kwargs, bad_min_degree):

        # min_degree lower bound of 1 is hard coded, so 0 is bad

        _kwargs['min_degree'] = bad_min_degree

        with pytest.raises(ValueError):
            SlimPoly(**_kwargs).fit_transform(X_np)


    @pytest.mark.parametrize('good_min_degree', (2,3,4))
    def test_good_min_degree(self, X_pd, _kwargs, good_min_degree):

        _kwargs['min_degree'] = good_min_degree
        _kwargs['degree'] = good_min_degree + 1

        SlimPoly(**_kwargs).fit_transform(X_pd)
    # END min_degree ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    # keep ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('junk_keep',
        (-1, np.pi, True, False, None, [1,2], {1,2}, {1: 'a'}, lambda x: 'junk')
    )
    def test_junk_keep(self, X_np, _kwargs, junk_keep):

        _kwargs['keep'] = junk_keep

        with pytest.raises(TypeError):
            SlimPoly(**_kwargs).fit_transform(X_np)


    @pytest.mark.parametrize('bad_keep', ('rubbish', 'trash', 'garbage'))
    def test_bad_keep(self, X_np, _kwargs, bad_keep):

        _kwargs['keep'] = bad_keep

        with pytest.raises(ValueError):
            SlimPoly(**_kwargs).fit_transform(X_np)


    @pytest.mark.parametrize('good_keep', ('first', 'last', 'random'))
    def test_good_keep(self, X_pd, _columns, _kwargs, good_keep):

        _kwargs['keep'] = good_keep
        SlimPoly(**_kwargs).fit_transform(X_pd)

    # END keep ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # interaction_only ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('junk_interaction_only',
        (-2.7, -1, 0, 1, 2.7, None, 'junk', (0,1), [1,2], {'a':1}, lambda x: x)
    )
    def test_junk_interaction_only(self, X_np, _kwargs, junk_interaction_only):

        _kwargs['interaction_only'] = junk_interaction_only

        with pytest.raises(TypeError):
            SlimPoly(**_kwargs).fit_transform(X_np)


    @pytest.mark.parametrize('good_interaction_only', (True, False))
    def test_good_interaction_only(
        self, X_pd, _columns, _kwargs, good_interaction_only
    ):

        _kwargs['interaction_only'] = good_interaction_only

        SlimPoly(**_kwargs).fit_transform(X_pd)
    # END interaction_only ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    # scan_X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('junk_scan_X',
        (-2.7, -1, 0, 1, 2.7, None, 'junk', (0,1), [1,2], {'a':1}, lambda x: x)
    )
    def test_junk_scan_X(self, X_np, _kwargs, junk_scan_X):

        _kwargs['scan_X'] = junk_scan_X

        with pytest.raises(TypeError):
            SlimPoly(**_kwargs).fit_transform(X_np)


    @pytest.mark.parametrize('good_scan_X', (True, False))
    def test_good_scan_X(self, X_pd, _columns, _kwargs, good_scan_X):

        _kwargs['scan_X'] = good_scan_X

        SlimPoly(**_kwargs).fit_transform(X_pd)
    # END scan_X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # sparse_output ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    @pytest.mark.parametrize('junk_sparse_output',
        (-2.7, -1, 0, 1, 2.7, None, 'junk', (0,1), [1,2], {'a':1}, lambda x: x)
    )
    def test_junk_sparse_output(self, X_np, _kwargs, junk_sparse_output):

        _kwargs['sparse_output'] = junk_sparse_output

        with pytest.raises(TypeError):
            SlimPoly(**_kwargs).fit_transform(X_np)


    @pytest.mark.parametrize('good_sparse_output', (True, False))
    def test_good_sparse_output(
        self, X_pd, _columns, _kwargs, good_sparse_output
    ):

        _kwargs['sparse_output'] = good_sparse_output

        SlimPoly(**_kwargs).fit_transform(X_pd)
    # END sparse_output ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # feature_name_combiner ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    # can be Literal['as_indices', 'as_feature_names']
    # or Callable[[Sequence[str], tuple[int,...]], str]

    @pytest.mark.parametrize('junk_feature_name_combiner',
        (-2.7, -1, 0, 1, 2.7, True, False, None, (0,1), [1,2], {1,2}, {'a':1})
    )
    def test_junk_feature_name_combiner(
        self, X_np, _kwargs, junk_feature_name_combiner
    ):

        _kwargs['feature_name_combiner'] = junk_feature_name_combiner

        with pytest.raises(ValueError):
            SlimPoly(**_kwargs).fit_transform(X_np)


    @pytest.mark.parametrize('bad_feature_name_combiner',
        ('that', 'was', 'trash')
    )
    def test_bad_feature_name_combiner(
        self, X_np, _kwargs, bad_feature_name_combiner
    ):

        _kwargs['feature_name_combiner'] = bad_feature_name_combiner

        with pytest.raises(ValueError):
            SlimPoly(**_kwargs).fit_transform(X_np)


    @pytest.mark.parametrize('good_feature_name_combiner',
        ('as_indices', 'as_feature_names', lambda x, y: str(uuid.uuid4())[:5])
    )
    def test_good_feature_name_combiner(
        self, X_pd, _columns, _kwargs, good_feature_name_combiner
    ):

        _kwargs['feature_name_combiner'] = good_feature_name_combiner

        SlimPoly(**_kwargs).fit_transform(X_pd)
    # END feature_name_combiner ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    # equal_nan ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @pytest.mark.parametrize('junk_equal_nan',
        (-1, 0, 1, np.pi, None, 'trash', [1, 2], {1, 2}, {'a': 1}, lambda x: x)
    )
    def test_non_bool_equal_nan(self, X_np, _kwargs, junk_equal_nan):

        _kwargs['equal_nan'] = junk_equal_nan

        with pytest.raises(TypeError):
            SlimPoly(**_kwargs).fit_transform(X_np)


    @pytest.mark.parametrize('good_equal_nan', [True, False])
    def test_equal_nan_accepts_bool(self, X_np, _kwargs, good_equal_nan):

        _kwargs['equal_nan'] = good_equal_nan

        SlimPoly(**_kwargs).fit_transform(X_np)

    # END equal_nan ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # rtol & atol ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('_trial', ('rtol', 'atol'))
    @pytest.mark.parametrize('_junk',
        (None, 'trash', [1,2], {1,2}, {'a':1}, lambda x: x, min)
    )
    def test_junk_rtol_atol(self, X_np, _kwargs, _trial, _junk):

        _kwargs[_trial] = _junk

        # non-num are handled by np.allclose, let it raise
        # whatever it will raise
        with pytest.raises(Exception):
            SlimPoly(**_kwargs).fit_transform(X_np)


    @pytest.mark.parametrize('_trial', ('rtol', 'atol'))
    @pytest.mark.parametrize('_bad', [-np.pi, -2, -1, True, False])
    def test_bad_rtol_atol(self, X_np, _kwargs, _trial, _bad):

        _kwargs[_trial] = _bad

        with pytest.raises(ValueError):
            SlimPoly(**_kwargs).fit_transform(X_np)


    @pytest.mark.parametrize('_trial', ('rtol', 'atol'))
    @pytest.mark.parametrize('_good', (1e-5, 1e-6, 1e-1))
    def test_good_rtol_atol(self, X_np, _kwargs, _trial, _good):

        _kwargs[_trial] = _good

        SlimPoly(**_kwargs).fit_transform(X_np)

    # END rtol & atol ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # n_jobs ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('junk_n_jobs',
        (-2.7, 2.7, True, False, 'trash', [1, 2], {'a': 1}, lambda x: x, min)
    )
    def test_junk_n_jobs(self, X_np, _kwargs, junk_n_jobs):

        _kwargs['n_jobs'] = junk_n_jobs

        with pytest.raises(TypeError):
            SlimPoly(**_kwargs).fit_transform(X_np)


    @pytest.mark.parametrize('bad_n_jobs', [-3, -2, 0])
    def test_bad_n_jobs(self, X_np, _kwargs, bad_n_jobs):

        _kwargs['n_jobs'] = bad_n_jobs

        with pytest.raises(ValueError):
            SlimPoly(**_kwargs).fit_transform(X_np)


    @pytest.mark.parametrize('good_n_jobs', [-1, 1, 10, None])
    def test_good_n_jobs(self, X_np, _kwargs, good_n_jobs):

        _kwargs['n_jobs'] = good_n_jobs

        SlimPoly(**_kwargs).fit_transform(X_np)

    # END n_jobs ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

# END test input validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestX:

    # - only accepts ndarray, pd.DataFrame, and all ss
    # - cannot be None
    # - must be 2D
    # - must have at least 1 or 2 columns, depending on interaction_only
    # - must have at least 1 sample
    # - allows nan
    # - output is C contiguous
    # - num columns must equal num columns seen during fit
    # - validates all instance attrs --- not tested here, see _validation
    # - does not mutate X


    # CONTAINERS #######################################################
    def test_excepts_anytime_x_is_none(self, X_np, _kwargs):

        # this is handled by _val_X

        with pytest.raises(ValueError):
            SlimPoly(**_kwargs).fit(None)

        with pytest.raises(ValueError):
            SlimPoly(**_kwargs).partial_fit(None)

        with pytest.raises(ValueError):
            TestCls = SlimPoly(**_kwargs)
            TestCls.fit(X_np)
            TestCls.transform(None)
            del TestCls

        with pytest.raises(ValueError):
            SlimPoly(**_kwargs).fit_transform(None)


    @pytest.mark.parametrize('_junk_X',
        (-1, 0, 1, 3.14, None, 'junk', [0, 1], (1,), {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_X(self, _junk_X, _kwargs, X_np):
        # this is being caught by validate_data at the top of partial_fit

        # PARTIAL_FIT
        with pytest.raises(ValueError):
            SlimPoly(**_kwargs).partial_fit(_junk_X)

        # TRANSFORN
        _SPF = SlimPoly(**_kwargs)
        _SPF.fit(X_np)
        # this is being caught by _val_X in _validation at the top of transform
        with pytest.raises(ValueError):
            _SPF.transform(_junk_X)


    @pytest.mark.parametrize('_format', ('py_list', 'py_tuple'))
    def test_rejects_invalid_container(self, X_np, _columns, _kwargs, _format):
        _SPF = SlimPoly(**_kwargs)

        if _format == 'py_list':
            _X_wip = list(map(list, X_np.copy()))
        elif _format == 'py_tuple':
            _X_wip = tuple(map(tuple, X_np.copy()))
        else:
            raise Exception

        # FROM PARTIAL_FIT
        if _format == 'py_list':
            with pytest.raises(ValueError):
                _SPF.partial_fit(_X_wip)
        elif _format == 'py_tuple':
            with pytest.raises(ValueError):
                _SPF.partial_fit(_X_wip)
        else:
            raise Exception
        # END PARTIAL_FIT

        # TRANSFORM
        if _format == 'py_list':
            with pytest.raises(ValueError):
                _SPF.transform(_X_wip)
        elif _format == 'py_tuple':
            with pytest.raises(ValueError):
                _SPF.transform(_X_wip)
        else:
            raise Exception
        # END TRANSFORM

    @pytest.mark.parametrize('_format',
         (
             'np', 'pd', 'pl', 'csr_matrix', 'csc_matrix', 'coo_matrix', 'dia_matrix',
             'lil_matrix', 'dok_matrix', 'bsr_matrix', 'csr_array', 'csc_array',
             'coo_array', 'dia_array', 'lil_array', 'dok_array', 'bsr_array'
         )
     )
    def test_X_container(self, X_np, _columns, _kwargs, _format):
        _X = X_np.copy()

        if _format == 'np':
            _X_wip = _X
        elif _format == 'pd':
            _X_wip = pd.DataFrame(data=_X, columns=_columns)
        elif _format == 'pl':
            _X_wip = pl.from_numpy(data=_X, schema=list(_columns))
        elif _format == 'csr_matrix':
            _X_wip = ss._csr.csr_matrix(_X)
        elif _format == 'csc_matrix':
            _X_wip = ss._csc.csc_matrix(_X)
        elif _format == 'coo_matrix':
            _X_wip = ss._coo.coo_matrix(_X)
        elif _format == 'dia_matrix':
            _X_wip = ss._dia.dia_matrix(_X)
        elif _format == 'lil_matrix':
            _X_wip = ss._lil.lil_matrix(_X)
        elif _format == 'dok_matrix':
            _X_wip = ss._dok.dok_matrix(_X)
        elif _format == 'bsr_matrix':
            _X_wip = ss._bsr.bsr_matrix(_X)
        elif _format == 'csr_array':
            _X_wip = ss._csr.csr_array(_X)
        elif _format == 'csc_array':
            _X_wip = ss._csc.csc_array(_X)
        elif _format == 'coo_array':
            _X_wip = ss._coo.coo_array(_X)
        elif _format == 'dia_array':
            _X_wip = ss._dia.dia_array(_X)
        elif _format == 'lil_array':
            _X_wip = ss._lil.lil_array(_X)
        elif _format == 'dok_array':
            _X_wip = ss._dok.dok_array(_X)
        elif _format == 'bsr_array':
            _X_wip = ss._bsr.bsr_array(_X)
        else:
            raise Exception

        try:
            _X_wip_before_partial_fit = _X_wip.copy()
        except:
            _X_wip_before_partial_fit = _X_wip.clone()

        # PARTIAL_FIT
        SlimPoly(**_kwargs).partial_fit(_X_wip)

        # verify _X_wip does not mutate in partial_fit()
        assert isinstance(_X_wip, type(_X_wip_before_partial_fit))
        assert _X_wip.shape == _X_wip_before_partial_fit.shape
        if isinstance(_X_wip, np.ndarray):
            assert _X_wip.flags['C_CONTIGUOUS'] is True

        if hasattr(_X_wip_before_partial_fit, 'toarray'):
            assert np.array_equal(
                _X_wip.toarray(),
                _X_wip_before_partial_fit.toarray()
            )
        elif isinstance(_X_wip_before_partial_fit, (pd.core.frame.DataFrame, pl.DataFrame)):
            assert _X_wip.equals(_X_wip_before_partial_fit)
        else:
            assert np.array_equal(_X_wip_before_partial_fit, _X_wip)
        # END PARTIAL_FIT

        # TRANSFORM
        _SPF = SlimPoly(**_kwargs)
        _SPF.fit(_X)  # fit on numpy, not the converted data

        try:
            _X_wip_before_transform = _X_wip.copy()
        except:
            _X_wip_before_transform = _X_wip.clone()

        out = _SPF.transform(_X_wip)
        assert isinstance(out, type(_X_wip))

        # if output is numpy, order is C
        if isinstance(out, np.ndarray):
            assert out.flags['C_CONTIGUOUS'] is True

        # verify _X_wip does not mutate in transform()
        assert isinstance(_X_wip, type(_X_wip_before_transform))
        assert _X_wip.shape == _X_wip_before_transform.shape

        if hasattr(_X_wip_before_transform, 'toarray'):
            assert np.array_equal(
                _X_wip.toarray(),
                _X_wip_before_transform.toarray()
            )
        elif isinstance(_X_wip_before_transform, (pd.core.frame.DataFrame, pl.DataFrame)):
            assert _X_wip.equals(_X_wip_before_transform)
        else:
            assert np.array_equal(_X_wip_before_transform, _X_wip)
        # END TRANSFORM
    # END CONTAINERS ###################################################


    # SHAPE ############################################################
    def test_rejects_no_samples(self, X_np, _kwargs, _columns):
        _X = X_np.copy()

        # PARTIAL_FIT
        # dont know what is actually catching this! maybe _validate_data?
        with pytest.raises(ValueError):
            SlimPoly(**_kwargs).partial_fit(
                np.empty((0, _X.shape[1]), dtype=np.float64)
            )
        # END PARTIAL_FIT

        # TRANSFORM
        _SPF = SlimPoly(**_kwargs)
        _SPF.fit(X_np)

        # this is caught by if _X.shape[0] == 0 in _val_X
        with pytest.raises(ValueError):
            _SPF.transform(
                np.empty((0, X_np.shape[1]), dtype=np.float64)
            )
        # END TRANSFORM

    def test_rejects_1D(self, X_np, _kwargs):

        with pytest.raises(ValueError):
            SlimPoly(**_kwargs).partial_fit(X_np[:, 0])

        _SPF = SlimPoly(**_kwargs)
        _SPF.fit(X_np)

        with pytest.raises(ValueError):
            _SPF.transform(X_np[:, 0])


    @pytest.mark.parametrize('_fst_fit_x_format', ('numpy', 'pandas'))
    @pytest.mark.parametrize('_fst_fit_x_hdr', (True, None))
    @pytest.mark.parametrize('_intx_only', (True, False))
    def test_X_as_single_column(
            self, _kwargs, _columns, X_np, _fst_fit_x_format, _fst_fit_x_hdr,
            _intx_only
    ):
        # DEPENDS ON interaction_only. IF True, MUST BE 2+ COLUMNS; IF False, CAN
        # BE 1 COLUMN.

        # y is ignored

        _kwargs['interaction_only'] = _intx_only

        VECTOR_X = X_np[:, 0].copy().reshape((-1, 1))

        if _fst_fit_x_format == 'numpy':
            if _fst_fit_x_hdr:
                pytest.skip(reason=f"numpy cannot have header")
            else:
                _fst_fit_X = VECTOR_X

        elif _fst_fit_x_format == 'pandas':
            if _fst_fit_x_hdr:
                _fst_fit_X = pd.DataFrame(data=VECTOR_X, columns=_columns[:1])
            else:
                _fst_fit_X = pd.DataFrame(data=VECTOR_X)

        else:
            raise Exception

        # PARTIAL_FIT
        if _intx_only:
            with pytest.raises(ValueError):
                SlimPoly(**_kwargs).fit_transform(_fst_fit_X)
        elif not _intx_only:
            out = SlimPoly(**_kwargs).fit_transform(_fst_fit_X)
            assert isinstance(out, type(_fst_fit_X))
        # END PARTIAL_FIT

        # TRANSFORM
        # test_X_num_columns(self)
        # this is dictated by partial_fit. partial_fit requires at least 1 or
        # 2 columns, and transform must have same number of features as fit
        if _intx_only:
            with pytest.raises(ValueError):
                SlimPoly(**_kwargs).fit_transform(_fst_fit_X)
        elif not _intx_only:
            out = SlimPoly(**_kwargs).fit_transform(_fst_fit_X)
            assert isinstance(out, type(_fst_fit_X))
        # END TRANSFORM


    @pytest.mark.parametrize('_fst_fit_x_format', ('numpy', 'pandas'))
    @pytest.mark.parametrize('_fst_fit_x_hdr', (True, None))
    def test_X_as_two_columns(
            self, _kwargs, _columns, X_np, _fst_fit_x_format, _fst_fit_x_hdr
    ):
        TWO_COLUMN_X = X_np[:, :2].copy().reshape((-1, 2))

        if _fst_fit_x_format == 'numpy':
            if _fst_fit_x_hdr:
                pytest.skip(reason=f"numpy cannot have header")
            else:
                _fst_fit_X = TWO_COLUMN_X

        elif _fst_fit_x_format == 'pandas':
            if _fst_fit_x_hdr:
                _fst_fit_X = pd.DataFrame(data=TWO_COLUMN_X, columns=_columns[:2])
            else:
                _fst_fit_X = pd.DataFrame(data=TWO_COLUMN_X)

        else:
            raise Exception

        out = SlimPoly(**_kwargs).fit_transform(_fst_fit_X)
        assert isinstance(out, type(_fst_fit_X))

        out = SlimPoly(**_kwargs).fit_transform(_fst_fit_X)
        assert isinstance(out, type(_fst_fit_X))


    def test_rejects_bad_num_features(self, X_np, _kwargs, _columns):
        # SHOULD RAISE ValueError WHEN COLUMNS DO NOT EQUAL NUMBER OF
        # FITTED COLUMNS

        _SPF = SlimPoly(**_kwargs)
        _SPF.fit(X_np)

        __ = np.array(_columns)
        for obj_type in ['np', 'pd', 'pl']:
            for diff_cols in ['more', 'less', 'same']:
                if diff_cols == 'same':
                    _X = X_np.copy()
                    if obj_type == 'pd':
                        _X = pd.DataFrame(data=_X, columns=__)
                    elif obj_type == 'pl':
                        _X = pl.from_numpy(data=_X, schema=list(__))
                elif diff_cols == 'less':
                    _X = X_np[:, :-1].copy()
                    if obj_type == 'pd':
                        _X = pd.DataFrame(data=_X, columns=__[:-1])
                    elif obj_type == 'pl':
                        _X = pl.from_numpy(data=_X, schema=list(__[:-1]))
                elif diff_cols == 'more':
                    _X = np.hstack((X_np.copy(), X_np.copy()))
                    _COLUMNS = np.hstack((__, np.char.upper(__)))
                    if obj_type == 'pd':
                        _X = pd.DataFrame(data=_X, columns=_COLUMNS)
                    elif obj_type == 'pl':
                        _X = pl.from_numpy(data=_X, schema=list(_COLUMNS))
                else:
                    raise Exception

                if diff_cols == 'same':
                    _SPF.transform(_X)
                else:
                    with pytest.raises(ValueError):
                        _SPF.transform(_X)

        del _SPF, obj_type, diff_cols, _X


    @pytest.mark.parametrize('fst_fit_name', ['DF1', 'DF2', 'NO_HDR_DF'])
    @pytest.mark.parametrize('scd_fit_name', ['DF1', 'DF2', 'NO_HDR_DF'])
    @pytest.mark.parametrize('trfm_name', ['DF1', 'DF2', 'NO_HDR_DF'])
    def test_except_or_warn_on_different_headers(
        self, _X_factory, _kwargs, _columns, fst_fit_name, scd_fit_name,
        trfm_name, _shape
    ):

        # TEST ValueError WHEN SEES A DF HEADER DIFFERENT FROM FIRST-SEEN HEADER

        # np.flip(_columns) is bad columns
        _col_dict = {'DF1': _columns, 'DF2': np.flip(_columns), 'NO_HDR_DF': None}

        TestCls = SlimPoly(**_kwargs)
        # pizza polars?
        _factory_kwargs = {
            '_dupl':None, '_format':'pd', '_dtype':'flt', '_has_nan':False,
            '_constants': None, '_shape':_shape
        }

        fst_fit_X = _X_factory(_columns=_col_dict[fst_fit_name], **_factory_kwargs)
        scd_fit_X = _X_factory(_columns=_col_dict[scd_fit_name], **_factory_kwargs)
        trfm_X = _X_factory(_columns=_col_dict[trfm_name], **_factory_kwargs)

        _objs = [fst_fit_name, scd_fit_name, trfm_name]
        # EXCEPT IF 2 DIFFERENT HEADERS ARE SEEN
        pybear_exception = 0
        pybear_exception += bool('DF1' in _objs and 'DF2' in _objs)
        # IF FIRST FIT WAS WITH NO HEADER, THEN ANYTHING GETS THRU ON
        # SUBSEQUENT partial_fits AND transform
        pybear_exception -= bool(fst_fit_name == 'NO_HDR_DF')
        pybear_exception = max(0, pybear_exception)

        # WARN IF HAS-HEADER AND NOT-HEADER BOTH PASSED DURING fits/transform
        pybear_warn = 0
        if not pybear_exception:
            pybear_warn += ('NO_HDR_DF' in _objs and 'NO_HDR_DF' in _objs)
            # IF NONE OF THEM HAD A HEADER, THEN NO WARNING
            pybear_warn -= ('DF1' not in _objs and 'DF2' not in _objs)
            pybear_warn = max(0, pybear_warn)

        del _objs

        if pybear_exception:
            with pytest.raises(Exception):
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

    # END SHAPE ########################################################


@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestPartialFit:

    #     def partial_fit(
    #         self,
    #         X: XContainer,
    #         y: any=None
    #     ) -> Self:


    @pytest.mark.parametrize('_format', ('np', 'pd', 'pl', 'bsr_array'))
    def test_X_is_not_mutated(
        self, _X_factory, _columns, _shape, _kwargs, _format
    ):
        pass


    @pytest.mark.parametrize('_test_y',
        (-1, 0, 1, np.pi, True, False, None, 'trash', [1, 2], {1, 2}, {'a': 1},
        lambda x: x, min)
    )
    def test_fit_partial_fit_accept_Y_equals_anything(self, _kwargs, X_np, _test_y):
        SlimPoly(**_kwargs).fit(X_np, _test_y)
        SlimPoly(**_kwargs).partial_fit(X_np, _test_y)


    def test_conditional_access_to_partial_fit_and_fit(
        self, X_np, _kwargs
    ):

        TestCls = SlimPoly(**_kwargs)

        # 1) partial_fit() should allow unlimited number of subsequent partial_fits()
        for _ in range(5):
            TestCls.partial_fit(X_np)

        del TestCls

        # 2) one call to fit() should allow subsequent attempts to partial_fit()
        TestCls = SlimPoly(**_kwargs)
        TestCls.fit(X_np)
        TestCls.partial_fit(X_np)

        del TestCls

        # 3) one call to fit() should allow later attempts to fit() (2nd fit will reset)
        TestCls = SlimPoly(**_kwargs)
        TestCls.fit(X_np)
        TestCls.fit(X_np)

        del TestCls

        # 4) calls to partial_fit() should allow later attempt to fit() (fit will reset)
        TestCls = SlimPoly(**_kwargs)
        TestCls.partial_fit(X_np)
        TestCls.fit(X_np)

        # 5) fit transform should allow calls ad libido
        for _ in range(5):
            TestCls.fit_transform(X_np)

        del TestCls


    @pytest.mark.parametrize('_keep', ('first', 'last', 'random'))
    def test_many_partial_fits_equal_one_big_fit(self, _kwargs, _keep):

        # **** **** **** **** **** **** **** **** **** **** **** **** ****
        # THIS TEST IS CRITICAL FOR VERIFYING THAT transform PULLS THE
        # SAME COLUMN INDICES FOR ALL CALLS TO transform() WHEN
        # keep=='random'
        # **** **** **** **** **** **** **** **** **** **** **** **** ****

        _kwargs['keep'] = _keep

        # rig X to have columns that will create duplicates when expanded
        _X = np.array(
            [
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 0]
            ], dtype=np.uint8
        )


        # ** ** ** ** ** ** ** ** ** ** **
        # TEST THAT ONE-SHOT partial_fit/transform == ONE-SHOT fit/transform
        OneShotPartialFitTestCls = SlimPoly(**_kwargs)
        OneShotPartialFitTestCls.partial_fit(_X)

        OneShotFullFitTestCls = SlimPoly(**_kwargs)
        OneShotFullFitTestCls.fit(_X)

        if _keep != 'random':
            _ = OneShotPartialFitTestCls.poly_combinations_
            __ = OneShotFullFitTestCls.poly_combinations_
            assert _ == __
            del _, __

        # this should be true for all 'keep', including random
        # (random too only because of the special design of X)
        assert np.array_equal(
            OneShotPartialFitTestCls.transform(_X),
            OneShotFullFitTestCls.transform(_X)
        ), f"one shot partial fit trfm X != one shot full fit trfm X"

        # del ONE_SHOT_PARTIAL_FIT_TRFM_X, ONE_SHOT_FULL_FIT_TRFM_X
        del OneShotPartialFitTestCls, OneShotFullFitTestCls

        # END TEST THAT ONE-SHOT partial_fit/transform==ONE-SHOT fit/transform
        # ** ** ** ** ** ** ** ** ** ** **


        # ** ** ** ** ** ** ** ** ** ** **
        # TEST PARTIAL FIT KEPT COMBINATIONS ARE THE SAME WHEN FULL DATA
        # IS partial_fit() 2X
        # SlimPoly should cause the same columns to be kept
        # when the same data is partial_fit multiple times
        SingleFitTestClass = SlimPoly(**_kwargs)
        SingleFitTestClass.fit(_X)

        DoublePartialFitTestClass = SlimPoly(**_kwargs)
        DoublePartialFitTestClass.partial_fit(_X)
        DoublePartialFitTestClass.partial_fit(_X)

        if _keep != 'random':
            _ = SingleFitTestClass.poly_combinations_
            __ = DoublePartialFitTestClass.poly_combinations_
            ___ = DoublePartialFitTestClass.poly_combinations_

            assert _ == __
            assert _ == ___

            del _, __, ___


        assert np.array_equal(
            SingleFitTestClass.transform(_X),
            DoublePartialFitTestClass.transform(_X)
        )

        del SingleFitTestClass, DoublePartialFitTestClass

        # END PARTIAL FIT CONSTANTS ARE THE SAME WHEN FULL DATA IS partial_fit() 2X
        # ** ** ** ** ** ** ** ** ** ** **

        # ** ** ** ** ** ** ** ** ** ** **# ** ** ** ** ** ** ** ** ** ** **
        # ** ** ** ** ** ** ** ** ** ** **# ** ** ** ** ** ** ** ** ** ** **
        # TEST MANY PARTIAL FITS == ONE BIG FIT

        # STORE CHUNKS TO ENSURE THEY STACK BACK TO THE ORIGINAL X
        _chunks = 3
        X_CHUNK_HOLDER = []
        for row_chunk in range(_chunks):
            _mask_start = row_chunk * _X.shape[0] // _chunks
            _mask_end = (row_chunk + 1) * _X.shape[0] // _chunks
            X_CHUNK_HOLDER.append(_X[_mask_start:_mask_end, :])
        del _mask_start, _mask_end

        assert np.array_equiv(np.vstack(X_CHUNK_HOLDER), _X), \
            f"agglomerated X chunks != original X"

        PartialFitTestCls = SlimPoly(**_kwargs)
        OneShotFitTransformTestCls = SlimPoly(**_kwargs)

        # PIECEMEAL PARTIAL FIT
        for X_CHUNK in X_CHUNK_HOLDER:
            PartialFitTestCls.partial_fit(X_CHUNK)

        # PIECEMEAL TRANSFORM ******************************************
        # THIS CANT BE UNDER THE partial_fit LOOP, ALL FITS MUST BE COMPLETED
        # BEFORE DOING ANY TRFMS
        PARTIAL_TRFM_X_HOLDER = []
        for X_CHUNK in X_CHUNK_HOLDER:

            PARTIAL_TRFM_X_HOLDER.append(
                PartialFitTestCls.transform(X_CHUNK)
            )

        # AGGLOMERATE PARTIAL TRFMS FROM PARTIAL FIT
        FULL_TRFM_X_FROM_PARTIAL_FIT_PARTIAL_TRFM = \
            np.vstack(PARTIAL_TRFM_X_HOLDER)
        # END PIECEMEAL TRANSFORM **************************************

        # DO ONE-SHOT TRANSFORM OF X ON THE PARTIALLY FIT INSTANCE
        FULL_TRFM_X_FROM_PARTIAL_FIT_ONESHOT_TRFM = \
            PartialFitTestCls.transform(_X)

        del PartialFitTestCls


        # ONE-SHOT FIT TRANSFORM
        FULL_TRFM_X_ONE_SHOT_FIT_TRANSFORM = \
            OneShotFitTransformTestCls.fit_transform(_X)

        del OneShotFitTransformTestCls

        # ASSERT ALL AGGLOMERATED X TRFMS ARE EQUAL
        assert np.array_equiv(
                FULL_TRFM_X_ONE_SHOT_FIT_TRANSFORM,
                FULL_TRFM_X_FROM_PARTIAL_FIT_PARTIAL_TRFM
            ), \
            (f"compiled trfm X from partial fit / partial trfm != "
             f"one-shot fit/trfm X")

        assert np.array_equiv(
                FULL_TRFM_X_ONE_SHOT_FIT_TRANSFORM,
                FULL_TRFM_X_FROM_PARTIAL_FIT_ONESHOT_TRFM
            ), (f"trfm X from partial fits / one-shot trfm != one-shot "
                f"fit/trfm X")

        # TEST MANY PARTIAL FITS == ONE BIG FIT
        # ** ** ** ** ** ** ** ** ** ** **# ** ** ** ** ** ** ** ** ** ** **
        # ** ** ** ** ** ** ** ** ** ** **# ** ** ** ** ** ** ** ** ** ** **


@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestTransform:

    #     def transform(
    #         self,
    #         X: XContainer,
    #         *,
    #         copy: bool = None
    #     ) -> XContainer:


    @pytest.mark.parametrize('_copy',
        (-1, 0, 1, 3.14, True, False, None, 'junk', [0, 1], (1,), {'a': 1}, min)
    )
    def test_copy_validation(self, X_np, _shape, _kwargs, _copy):
        pass


    @pytest.mark.parametrize('x_input_type', ['np_array', 'pandas', 'scipy_sparse_csc'])
    @pytest.mark.parametrize('output_type', [None, 'default', 'pandas', 'polars'])
    def test_output_types(
        self, X_np, _columns, _kwargs, x_input_type, output_type
    ):

        if x_input_type == 'np_array':
            _X = X_np
        elif x_input_type == 'pandas':
            _X = pd.DataFrame(data=X_np, columns=_columns)
        elif x_input_type == 'scipy_sparse_csc':
            _X = ss.csc_array(X_np)
        else:
            raise Exception

        TestCls = SlimPoly(**_kwargs)
        TestCls.set_output(transform=output_type)

        TRFM_X = TestCls.fit_transform(_X)

        # if output_type is None, should return same type as given
        if output_type is None:
            assert type(TRFM_X) == type(_X), \
                (f"output_type is None, X output type ({type(TRFM_X)}) != "
                 f"X input type ({type(_X)})")
        # if output_type is 'default', should return np array no matter what given
        elif output_type == 'default':
            assert isinstance(TRFM_X, np.ndarray), \
                f"output_type is default, TRFM_X is {type(TRFM_X)}"
        # if output_type is 'pandas', should return pd df no matter what given
        elif output_type == 'pandas':
            # pandas.core.frame.DataFrame
            assert isinstance(TRFM_X, pd.core.frame.DataFrame), \
                f"output_type is pandas dataframe, TRFM_X is {type(TRFM_X)}"
        elif output_type == 'polars':
            # polars.dataframe.frame.DataFrame
            assert isinstance(TRFM_X, pl.dataframe.frame.DataFrame), \
                f"output_type is polars dataframe, TRFM_X is {type(TRFM_X)}"
        else:
            raise Exception


    @pytest.mark.parametrize('_format', ('np', 'pd', 'pl', 'coo_matrix'))
    def test_X_is_not_mutated(
        self, _X_factory, _columns, _shape, _kwargs, _format
    ):

        _X_wip = _X_factory(
            _has_nan=False, _format=_format, _dtype='flt',
            _columns=_columns, _constants=None, _noise=0, _zeros=None,
            _shape=_shape
        )

        if _format == 'np':
            assert _X_wip.flags['C_CONTIGUOUS'] is True

        try:
            _X_wip_before = _X_wip.copy()
        except:
            _X_wip_before = _X_wip.clone()


        _SPF = SlimPoly(**_kwargs).fit(_X_wip)

        # verify _X_wip does not mutate in transform()
        TRFM_X = _SPF.transform(_X_wip)


        # ASSERTIONS v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v
        assert isinstance(_X_wip, type(_X_wip_before))
        assert _X_wip.shape == _X_wip_before.shape

        if isinstance(_X_wip_before, np.ndarray):
            assert np.array_equal(_X_wip_before, _X_wip, equal_nan=True)
            assert _X_wip.flags['C_CONTIGUOUS'] is True
        elif hasattr(_X_wip_before, 'columns'):    # DATAFRAMES
            assert _X_wip.equals(_X_wip_before)
        elif hasattr(_X_wip_before, 'toarray'):
            assert np.array_equal(
                _X_wip.toarray(), _X_wip_before.toarray(), equal_nan=True
            )
        else:
            raise Exception


    def test_all_ones_in_X(self, _kwargs, X_np):

        # this should always end in a degenerate state that causes no-ops
        # this tests whether SPF can handle the intermediate degenerate
        # states that exist when the constant in X is 1.

        # do this with interaction_only = False
        _kwargs['interaction_only'] = False

        # scan_X must be True! or SlimPoly wont find the constants in X
        _kwargs['scan_X'] = True

        # create 2 columns, one of them is all ones

        # SlimPoly should see the ones as a column of constants
        # in X, and transform() should always be a no-op

        _X = X_np[:, :2].copy()

        # set a column to all ones
        _X[:, -1] = 1

        TestCls = SlimPoly(**_kwargs).fit(_X)

        with pytest.warns():
            out = TestCls.transform(_X)
        assert out is None


    def test_all_zeros_in_X(self, _kwargs, X_np):

        # this should always end in a degenerate state that causes no-ops
        # this tests whether SPF can handle the intermediate degenerate
        # states that exist when the constant in X is 0.

        # do this with interaction_only = False
        _kwargs['interaction_only'] = False

        # scan_X must be True! or SlimPoly wont find the constants in X
        _kwargs['scan_X'] = True

        # create 2 columns, one of them is all zeros

        # SlimPoly should see the zeros as a column of constants
        # in X, and transform() should always be a no-op

        _X = X_np[:, :2].copy()

        # set a column to all zeros
        _X[:, -1] = 0

        TestCls = SlimPoly(**_kwargs).fit(_X)

        with pytest.warns():
            out = TestCls.transform(_X)
        assert out is None


    @pytest.mark.parametrize('_equal_nan', (True, False))
    def test_one_all_nans(self, _kwargs, X_np, _shape, _equal_nan):

        # do this with interaction_only = False
        _kwargs['interaction_only'] = False

        # scan_X must be True! or SlimPoly wont find the constants in X
        _kwargs['scan_X'] = True

        _kwargs['equal_nan'] = _equal_nan

        # create 2 columns, one of them is all nans

        # if equal_nan, then SlimPoly should see the nans as a column of
        # constants in X, and transform() should always be a no-op

        # if not equal_nan, then it wont be a column of constants in X,
        # the poly component should have 3 columns.
        #   1) column 1 squared
        #   2) column 2 squared, which should be nans
        #   3) the interaction term, and that should be all nans
        # should return the original 2 columns and the 3 poly columns

        _X = X_np[:, :2].copy()

        # set a column to all nans
        _X[:, -1] = np.nan
        # verify last column is all nans
        assert all(nan_mask_numerical(_X[:, -1]))

        TestCls = SlimPoly(**_kwargs).fit(_X)

        if _equal_nan:
            with pytest.warns():
                out = TestCls.transform(_X)
            assert out is None
        elif not _equal_nan:
            out = TestCls.transform(_X)

            assert out.shape == (_shape[0], 5)

            assert np.array_equal(out[:, 0], _X[:, 0])
            assert all(nan_mask_numerical(out[:, 1]))
            assert np.array_equal(out[:, 2], np.power(_X[:, 0], 2), equal_nan=True)
            assert all(nan_mask_numerical(out[:, 3]))
            assert all(nan_mask_numerical(out[:, 4]))


    @pytest.mark.parametrize('_equal_nan', (True, False))
    def test_intx_creates_all_nans(self, _kwargs, _equal_nan):

        # rig X to have 2 columns that multiply to all nans

        _X = np.array(
            [
                [1, np.nan],
                [2, np.nan],
                [np.nan, 3],
                [np.nan, 4],
            ],
            dtype=np.float64
        )

        # do this with interaction_only = True
        _kwargs['interaction_only'] = True

        _kwargs['equal_nan'] = _equal_nan

        # if equal_nan, then SlimPoly should see the output as a column of
        # constants and not append it.

        # if not equal_nan, then it wont be a column of constants in POLY,
        # the expansion should be 1 column, a column of all nans.
        # should return the original 2 columns and the 1 poly

        TestCls = SlimPoly(**_kwargs).fit(_X)

        if _equal_nan:
            out = TestCls.transform(_X)
            assert np.array_equal(out, _X, equal_nan=True)
        elif not _equal_nan:
            out = TestCls.transform(_X)

            assert out.shape == (_X.shape[0], 3)

            assert np.array_equal(out[:, 0], _X[:, 0], equal_nan=True)
            assert np.array_equal(out[:, 1], _X[:, 1], equal_nan=True)
            assert all(nan_mask_numerical(out[:, 2]))



    # pizza this needs a lot of work for polars. assess the need for this test.
    @pytest.mark.parametrize('_format', ('np', 'pd', 'csr'))   # , 'pl'
    @pytest.mark.parametrize('_dtype',
         ('int8', 'int16', 'int32', 'int64', 'float64', '<U10', 'object')
    )
    @pytest.mark.parametrize('_has_nan', (True, False))
    @pytest.mark.parametrize('_min_degree', (1, 2))
    def test_dtype_handling(
        self, _format, _dtype, _has_nan, _min_degree, X_np, _shape, _columns,
        _kwargs
    ):

        # poly is always created as float64 and if/when merged, the
        # original data is also converted to float64.

        # _dtype '<U10' and 'object' test when numerical data is passed
        # with these formats

        # degree is always set to 2
        # when min_degree == 1, tests mashup of original data and poly
        # when min_degree == 2, tests poly only

        # skip impossible scenarios
        if _has_nan and _dtype in ('int8', 'int16', 'int32', 'int64'):
            pytest.skip(reason='cant have nans in np int dtype')

        if _format == 'csr':
            if _dtype in ('<U10', 'object'):
                pytest.skip(
                    reason='cant have str or object dtype for scipy sparse'
                )
        # END skip impossible scenarios

        _dtype_dict = {
            'int8': np.int8, 'int16': np.int16, 'int32': np.int32,
            'int64': np.int64, 'float64': np.float64, '<U10': str,
            'object': object
        }

        _base_X = X_np.astype(_dtype_dict[_dtype])

        # pizza can _X_factory be used here?
        # build X - - - - - - - - - - - - - - - - - - - - - -
        if _format in ('np', 'csr'):
            _X = _base_X
            if _has_nan:
                for _c_idx in range(_shape[1]):
                    _rand_idxs = np.random.choice(
                        range(_shape[0]), _shape[0] // 5, replace=False
                    )
                    _X[_rand_idxs, _c_idx] = np.nan

                # verify nans were made correctly
                assert np.sum(nan_mask(_X)) > 0

            if _format == 'csr':
                _X = ss.csr_array(_base_X)

            assert _X.dtype is _base_X.dtype

        elif _format == 'pd':
            _X = pd.DataFrame(data=_base_X, columns=_columns)
            if _has_nan:
                for _column in _X.columns:
                    _rand_idxs = np.random.choice(
                        range(_shape[0]), _shape[0] // 5, replace=False
                    )
                    _X.loc[_rand_idxs, _column] = pd.NA

                # verify nans were made correctly
                assert np.sum(nan_mask(_X)) > 0

            for _pd_dtype in _X.dtypes:
                if _dtype == '<U10':
                    assert _pd_dtype == object
                else:
                    assert _pd_dtype == _dtype_dict[_dtype]

        elif _format == 'pl':
            _X = pl.from_numpy(data=_base_X, schema=list(_columns))
            # pizza figure this mess out
            # if _has_nan:
            #     for _column in _X.columns:
            #         _rand_idxs = np.random.choice(
            #             range(_shape[0]), _shape[0] // 5, replace=False
            #         )
            #         _X[_rand_idxs, _column] = None
            #
            #     # verify nans were made correctly
            #     assert np.sum(nan_mask(_X)) > 0
            #
            # for _pl_dtype in _X.dtypes:
            #     if _dtype == '<U10':
            #         assert _pl_dtype == pl.Object
            #     else:
            #         assert _pd_dtype == _dtype_dict[_dtype]

        else:
            raise Exception
        # END build X - - - - - - - - - - - - - - - - - - - -



        _kwargs['degree'] = 2
        _kwargs['min_degree'] = _min_degree
        _kwargs['scan_X'] = False
        _kwargs['sparse_output'] = False
        _kwargs['equal_nan'] = True

        TestCls = SlimPoly(**_kwargs)

        if _dtype == '<U10':
            # 25_03_09 no longer coercing numbers passed as str to float
            with pytest.raises(TypeError):
                TestCls.fit_transform(_X)
            pytest.skip(reason=f"cannot do more tests after except")
        else:
            out = TestCls.fit_transform(_X)


        if _format == 'pd':
            for _c_idx, _out_dtype in enumerate(out.dtypes):
                assert _out_dtype == np.float64
        elif _format == 'pl':
            for _c_idx, _out_dtype in enumerate(out.dtypes):
                assert _out_dtype == pl.Float64
        else:
            assert out.dtype == np.float64


    @pytest.mark.parametrize('x_input_type', ['np_array', 'pandas', 'scipy_sparse_csc'])
    @pytest.mark.parametrize('_sparse_output', (True, False))
    def test_sparse_output(
            self, X_np, _columns, _kwargs, x_input_type, _sparse_output
    ):

        _kwargs['sparse_output'] = _sparse_output

        if x_input_type == 'np_array':
            _X = X_np
        elif x_input_type == 'pandas':
            _X = pd.DataFrame(data=X_np, columns=_columns)
        elif x_input_type == 'scipy_sparse_csc':
            _X = ss.csc_array(X_np)
        else:
            raise Exception

        TestCls = SlimPoly(**_kwargs)

        out = TestCls.fit_transform(_X)

        # when 'sparse_output' is False, return in the original container
        # when True, always return as ss csr, no matter what input container
        if _sparse_output:
            assert isinstance(out, (ss.csr_matrix, ss.csr_array))
        elif not _sparse_output:
            assert isinstance(out, type(_X))


class TestFitTransform:


    @pytest.mark.parametrize('x_input_type', ['np_array', 'pandas', 'scipy_sparse_csc'])
    @pytest.mark.parametrize('output_type', [None, 'default', 'pandas', 'polars'])
    def test_output_types(
        self, X_np, _columns, _kwargs, x_input_type, output_type
    ):

        if x_input_type == 'np_array':
            _X = X_np
        elif x_input_type == 'pandas':
            _X = pd.DataFrame(data=X_np, columns=_columns)
        elif x_input_type == 'scipy_sparse_csc':
            _X = ss.csc_array(X_np)
        else:
            raise Exception

        TestCls = SlimPoly(**_kwargs)
        TestCls.set_output(transform=output_type)

        TRFM_X = TestCls.fit_transform(_X)

        # if output_type is None, should return same type as given
        if output_type is None:
            assert type(TRFM_X) == type(_X), \
                (f"output_type is None, X output type ({type(TRFM_X)}) != "
                 f"X input type ({type(_X)})")
        # if output_type is 'default', should return np array no matter what given
        elif output_type == 'default':
            assert isinstance(TRFM_X, np.ndarray), \
                f"output_type is default, TRFM_X is {type(TRFM_X)}"
        # if output_type is 'pandas', should return pd df no matter what given
        elif output_type == 'pandas':
            # pandas.core.frame.DataFrame
            assert isinstance(TRFM_X, pd.core.frame.DataFrame), \
                f"output_type is pandas dataframe, TRFM_X is {type(TRFM_X)}"
        elif output_type == 'polars':
            # polars.dataframe.frame.DataFrame
            assert isinstance(TRFM_X, pl.dataframe.frame.DataFrame), \
                f"output_type is polars dataframe, TRFM_X is {type(TRFM_X)}"
        else:
            raise Exception


    @pytest.mark.parametrize('x_input_type', ['np_array', 'pandas', 'scipy_sparse_csc'])
    @pytest.mark.parametrize('_sparse_output', (True, False))
    def test_sparse_output(
            self, X_np, _columns, _kwargs, x_input_type, _sparse_output
    ):

        _kwargs['sparse_output'] = _sparse_output

        if x_input_type == 'np_array':
            _X = X_np
        elif x_input_type == 'pandas':
            _X = pd.DataFrame(data=X_np, columns=_columns)
        elif x_input_type == 'scipy_sparse_csc':
            _X = ss.csc_array(X_np)
        else:
            raise Exception

        TestCls = SlimPoly(**_kwargs)

        out = TestCls.fit_transform(_X)

        # when 'sparse_output' is False, return in the original container
        # when True, always return as ss csr, no matter what input container
        if _sparse_output:
            assert isinstance(out, (ss.csr_matrix, ss.csr_array))
        elif not _sparse_output:
            assert isinstance(out, type(_X))





