# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from copy import deepcopy
import itertools

import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as ss

from pybear.preprocessing import ColumnDeduplicateTransformer as CDT

from pybear.base.exceptions import NotFittedError

from pybear.utilities import (
    nan_mask,
    nan_mask_numerical,
    nan_mask_string
)



bypass = False


# v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
# FIXTURES

@pytest.fixture(scope='module')
def _dupls(_shape):
    # _dupl must be intermingled like [[0,8],[1,9]], not [[0,1],[8,9]]
    # for TestManyPartialFitsEqualOneBigFit to catch 'random' putting
    # out different columns over a sequence of transforms
    return [[0,_shape[1]-2], [1, _shape[1]-1]]


@pytest.fixture(scope='module')
def _X_np(_X_factory, _dupls, _shape):
    return _X_factory(
        _dupl=_dupls,
        _has_nan=False,
        _dtype='flt',
        _shape=_shape
    )

# END fixtures
# v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^


# test input validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestInitValidation:


    # keep ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('junk_keep',
        (-1,0,1, np.pi, True, False, None, [1,2], {1,2}, {'a':1}, lambda x: x)
    )
    def test_junk_keep(self, _X_np, _kwargs, junk_keep):

        _kwargs['keep'] = junk_keep

        with pytest.raises(TypeError):
            CDT(**_kwargs).fit_transform(_X_np)


    @pytest.mark.parametrize('bad_keep',
        ('trash', 'garbage', 'waste')
    )
    def test_bad_keep(self, _X_np, _kwargs, bad_keep):

        _kwargs['keep'] = bad_keep

        with pytest.raises(ValueError):
            CDT(**_kwargs).fit_transform(_X_np)


    @pytest.mark.parametrize('good_keep',
        ('first', 'last', 'random')
    )
    def test_good_keep(self, _X_np, _kwargs, good_keep):

        _kwargs['keep'] = good_keep
        CDT(**_kwargs).fit_transform(_X_np)

    # END keep ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # do_not_drop ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('junk_dnd',
        (-1, 0, 1, np.pi, True, False, 'trash', {'a': 1}, lambda x: x, min)
    )
    def test_rejects_not_list_like_or_none(self, _kwargs, _X_np, junk_dnd):

        _kwargs['do_not_drop'] = junk_dnd
        with pytest.raises(TypeError):
            CDT(**_kwargs).fit_transform(_X_np)


    @pytest.mark.parametrize('bad_dnd',
        ([True, min, 3.14], [min, max, float], [2.718, 3.141, 8.834])
    )
    def test_rejects_bad_list(self, _X_np, _kwargs, bad_dnd):

        _kwargs['do_not_drop'] = bad_dnd
        with pytest.raises(TypeError):
            CDT(**_kwargs).fit_transform(_X_np)


    @pytest.mark.parametrize('_format', ('np', 'pd', 'pl'))
    def test_dnd_str_handing(
        self, _X_factory, _shape, _kwargs, _columns, _format
    ):

        assert _format in ('np', 'pd', 'pl')

        _X_wip = _X_factory(
            _format=_format,
            _columns=_columns if _format in ['pd', 'pl'] else None,
            _shape=_shape
        )

        _kwargs['conflict'] = 'ignore'

        _kwargs['do_not_drop'] = [v for i, v in enumerate(_columns) if i % 2 == 0]
        if _format == 'np':
            # rejects str when no header
            with pytest.raises(TypeError):
                CDT(**_kwargs).fit_transform(_X_wip)
        elif _format in ['pd', 'pl']:
            # accepts good str always
            _kwargs['do_not_drop'] = [v for i, v in enumerate(_columns) if i % 2 == 0]
            CDT(**_kwargs).fit_transform(_X_wip)

        _kwargs['do_not_drop'] = ['a', 'b']
        if _format == 'np':
            # rejects str when no header
            with pytest.raises(TypeError):
                CDT(**_kwargs).fit_transform(_X_wip)
        elif _format in ['pd', 'pl']:
            # rejects bad str when header
            with pytest.raises(ValueError):
                CDT(**_kwargs).fit_transform(_X_wip)


    @pytest.mark.parametrize('_format', ('np', 'pd', 'pl'))
    def test_dnd_int_none_handling(
        self, _X_factory, _shape, _kwargs, _columns, _format
    ):

        assert _format in ('np', 'pd', 'pl')

        _X_wip = _X_factory(
            _format=_format,
            _columns=_columns if _format in ['pd', 'pl'] else None,
            _shape=_shape
        )

        # accepts good int always
        _kwargs['do_not_drop'] = [0, 1]
        CDT(**_kwargs).fit_transform(_X_wip)

        # rejects bad int always - 1
        _kwargs['do_not_drop'] = [-1, 1]
        with pytest.raises(ValueError):
            CDT(**_kwargs).fit_transform(_X_wip)

        # rejects bad int always - 2
        _kwargs['do_not_drop'] = [0, _X_wip.shape[1]]
        with pytest.raises(ValueError):
            CDT(**_kwargs).fit_transform(_X_wip)

        # accepts None always
        _kwargs['do_not_drop'] = None
        CDT(**_kwargs).fit_transform(_X_wip)
    # END do_not_drop ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # conflict  ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    @pytest.mark.parametrize('junk_conflict',
        (-1, 0, np.pi, True, None, [1, 2], {1, 2}, {'a': 1}, lambda x: x, min)
    )
    def test_junk_conflict(self, _X_np, _kwargs, junk_conflict):

        _kwargs['conflict'] = junk_conflict

        with pytest.raises(TypeError):
            CDT(**_kwargs).fit_transform(_X_np)


    @pytest.mark.parametrize('bad_conflict', ('trash', 'garbage', 'waste'))
    def test_bad_conflict(self, _X_np, _kwargs, bad_conflict):

        _kwargs['conflict'] = bad_conflict

        with pytest.raises(ValueError):
            CDT(**_kwargs).fit_transform(_X_np)


    @pytest.mark.parametrize('good_conflict', ('raise', 'ignore'))
    def test_good_conflict(self, _X_np, _kwargs, good_conflict):

        _kwargs['conflict'] = good_conflict

        CDT(**_kwargs).fit_transform(_X_np)
    # END conflict ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # rtol & atol ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('_param', ('rtol', 'atol'))
    @pytest.mark.parametrize('_junk',
        (True, False, None, 'trash', [1,2], {1,2}, {'a':1}, lambda x: x, min)
    )
    def test_junk_rtol_atol(self, _X_np, _kwargs, _param, _junk):

        _kwargs[_param] = _junk

        # non-num are handled by np.allclose, let it raise
        # whatever it will raise
        with pytest.raises(Exception):
            CDT(**_kwargs).fit_transform(_X_np)


    @pytest.mark.parametrize('_param', ('rtol', 'atol'))
    @pytest.mark.parametrize('_bad', (-np.pi, -2, -1))
    def test_bad_rtol_atol(self, _X_np, _kwargs, _param, _bad):

        _kwargs[_param] = _bad

        with pytest.raises(ValueError):
            CDT(**_kwargs).fit_transform(_X_np)


    @pytest.mark.parametrize('_param', ('rtol', 'atol'))
    @pytest.mark.parametrize('_good', (1e-5, 1e-6, 1e-1, 1_000_000))
    def test_good_rtol_atol(self, _X_np, _kwargs, _param, _good):

        _kwargs[_param] = _good

        CDT(**_kwargs).fit_transform(_X_np)

    # END rtol & atol ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # equal_nan ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @pytest.mark.parametrize('_junk',
        (-1, 0, 1, np.pi, None, 'trash', [1, 2], {1, 2}, {'a': 1}, lambda x: x)
    )
    def test_non_bool_equal_nan(self, _X_np, _kwargs, _junk):

        _kwargs['equal_nan'] = _junk

        with pytest.raises(TypeError):
            CDT(**_kwargs).fit_transform(_X_np)


    @pytest.mark.parametrize('_equal_nan', [True, False])
    def test_equal_nan_accepts_bool(self, _X_np, _kwargs, _equal_nan):

        _kwargs['equal_nan'] = _equal_nan

        CDT(**_kwargs).fit_transform(_X_np)

    # END equal_nan ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # n_jobs ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('junk_n_jobs',
        (True, False, 'trash', [1, 2], {1, 2}, {'a': 1}, lambda x: x, min)
    )
    def test_junk_n_jobs(self, _X_np, _kwargs, junk_n_jobs):

        _kwargs['n_jobs'] = junk_n_jobs

        with pytest.raises(TypeError):
            CDT(**_kwargs).fit_transform(_X_np)


    @pytest.mark.parametrize('bad_n_jobs', [-2, 0])
    def test_bad_n_jobs(self, _X_np, _kwargs, bad_n_jobs):

        _kwargs['n_jobs'] = bad_n_jobs

        with pytest.raises(ValueError):
            CDT(**_kwargs).fit_transform(_X_np)


    @pytest.mark.parametrize('good_n_jobs', [-1, 1, 3, None])
    def test_good_n_jobs(self, _X_np, _kwargs, good_n_jobs):

        _kwargs['n_jobs'] = good_n_jobs

        CDT(**_kwargs).fit_transform(_X_np)

    # END n_jobs ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

# END test input validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestX:

    # - accepts ndarray, pd.DataFrame, pl.DataFrame, and all ss
    # - cannot be None
    # - must be 2D
    # - must have at least 2 columns
    # - must have at least 1 sample
    # - num columns must equal num columns in column_mask_ (pizza do any tests cover this?)
    # - allows nan
    # - output is C contiguous
    # - partial_fit/transform num columns must equal num columns seen during first fit


    # CONTAINERS #######################################################
    @pytest.mark.parametrize('_junk_X',
        (-1, 0, 1, 3.14, None, 'junk', [0, 1], (1,), {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_X(self, _kwargs, _X_np, _junk_X):

        TestCls = CDT(**_kwargs)

        # these are caught by base.validate_data.
        with pytest.raises(ValueError):
            TestCls.partial_fit(_junk_X)

        with pytest.raises(ValueError):
            TestCls.fit(_junk_X)

        with pytest.raises(ValueError):
            TestCls.fit_transform(_junk_X)

        TestCls.fit(_X_np)

        with pytest.raises(ValueError) as e:
            TestCls.transform(_junk_X)
        assert not isinstance(e.value, NotFittedError)

        with pytest.raises(ValueError) as e:
            TestCls.inverse_transform(_junk_X)
        assert not isinstance(e.value, NotFittedError)


    @pytest.mark.parametrize('_format', ('py_list', 'py_tuple'))
    def test_rejects_invalid_container(self, _X_np, _columns, _kwargs, _format):

        assert _format in ('py_list', 'py_tuple')

        TestCls = CDT(**_kwargs)

        if _format == 'py_list':
            _X_wip = list(map(list, _X_np.copy()))
        elif _format == 'py_tuple':
            _X_wip = tuple(map(tuple, _X_np.copy()))

        with pytest.raises(ValueError):
            TestCls.partial_fit(_X_wip)

        with pytest.raises(ValueError):
            TestCls.fit(_X_wip)

        with pytest.raises(ValueError):
            TestCls.fit_transform(_X_wip)

        TestCls.fit(_X_np) # fit on numpy, not the converted data

        with pytest.raises(ValueError) as e:
            TestCls.transform(_X_wip)
        assert not isinstance(e.value, NotFittedError)

        with pytest.raises(ValueError) as e:
            TestCls.inverse_transform(_X_wip)
        assert not isinstance(e.value, NotFittedError)


    @pytest.mark.parametrize('_format',
        ('np', 'pd', 'pl', 'coo_matrix', 'dok_array', 'bsr_array')
    )
    def test_good_X_container(
            self, _X_factory, _columns, _shape, _kwargs, _dupls, _format
    ):

        _X_wip = _X_factory(
            _dupl=_dupls, _has_nan=False, _format=_format, _dtype='flt',
            _columns=_columns, _constants=None, _noise=0, _zeros=None,
            _shape=_shape
        )


        _CDT = CDT(**_kwargs)

        _CDT.partial_fit(_X_wip)

        _CDT.fit(_X_wip)

        _CDT.fit_transform(_X_wip)

        TRFM_X = _CDT.transform(_X_wip)

        _CDT.inverse_transform(TRFM_X)

    # END CONTAINERS #######################################################


    # SHAPE #############################################################
    @pytest.mark.parametrize('_format', ('np', 'pd', 'pl'))
    def test_rejects_1D(self, _X_np, _kwargs, _format):

        # validation order is
        # 1) check_fitted (for transform & inv_transform)
        # 2) base.validate_data, which catches dim & min columns
        # 3) _check_n_features in transform
        # so for the fits, transform & inv_transform, 1D will always catch first

        _CDT = CDT(**_kwargs)

        if _format == 'np':
            _X_wip = _X_np[:, 0]
        elif _format == 'pd':
            _X_wip = pd.Series(_X_np[:, 0])
        elif _format == 'pl':
            _X_wip = pl.Series(_X_np[:, 0])
        else:
            raise Exception

        with pytest.raises(ValueError):
            _CDT.partial_fit(_X_wip)

        with pytest.raises(ValueError):
            _CDT.fit(_X_wip)

        with pytest.raises(ValueError):
            _CDT.fit_transform(_X_wip)

        _CDT.fit(_X_np)

        with pytest.raises(ValueError) as e:
            _CDT.transform(_X_wip)
        assert not isinstance(e.value, NotFittedError)

        with pytest.raises(ValueError) as e:
            _CDT.inverse_transform(_X_wip)
        assert not isinstance(e.value, NotFittedError)


    @pytest.mark.parametrize('_num_cols', (0, 1, 2))
    @pytest.mark.parametrize('_format', ('np', 'pd', 'pl', 'dia_matrix'))
    def test_X_2D_number_of_columns(
            self, _X_np, _shape, _kwargs, _columns, _format, _num_cols
    ):

        # validation order is
        # 1) check_fitted (for transform & inv_transform)
        # 2) base.validate_data, which catches dim & min columns
        # 3) _check_n_features in transform
        # so for the fits, transform & inv_transform, validate_data will catch
        # for inverse_transform min is 1 column, everything else is 2

        _base_X = _X_np[:, :_num_cols]
        if _format == 'np':
            _X_wip = _base_X
        elif _format == 'pd':
            _X_wip = pd.DataFrame(_base_X, columns=_columns[:_num_cols])
        elif _format == 'pl':
            _X_wip = pl.from_numpy(_base_X, schema=list(_columns[:_num_cols]))
        elif _format == 'dia_matrix':
            _X_wip = ss.csc_array(_base_X)
        else:
            raise Exception

        assert len(_X_wip.shape) == 2
        assert _X_wip.shape[1] == _num_cols

        _CDT = CDT(**_kwargs)

        # inverse_transform can take 1, everything else needs >= 2
        if _num_cols in [0, 1]:
            with pytest.raises(ValueError):
                _CDT.partial_fit(_X_wip)
            with pytest.raises(ValueError):
                _CDT.fit(_X_wip)
            with pytest.raises(ValueError):
                _CDT.fit_transform(_X_wip)
            _CDT.fit(_X_np)
            with pytest.raises(ValueError) as e:
                _CDT.transform(_X_wip)
            assert not isinstance(e.value, NotFittedError)
            if _num_cols == 0:
                _CDT.fit(_X_np)
                with pytest.raises(ValueError) as e:
                    _CDT.inverse_transform(_X_wip)
                assert not isinstance(e.value, NotFittedError)
            elif _num_cols == 1:
                try:
                    # when type is ndarray it wont allow create like this
                    # kick out and just hstack since _base_X is np
                    _new_X_wip = type(_X_wip)(np.hstack((_base_X, _base_X)))
                except:
                    _new_X_wip = np.hstack((_base_X, _base_X))
                _CDT.fit(_new_X_wip)
                TRFM_X = _CDT.transform(_new_X_wip)
                _CDT.inverse_transform(TRFM_X)
        else:
            _CDT.partial_fit(_X_wip)
            _CDT.fit(_X_wip)
            _CDT.fit_transform(_X_wip)
            _CDT.fit(_X_np[:, :_num_cols])  # fit the instance
            TRFM_X = _CDT.transform(_X_wip)
            _CDT.inverse_transform(TRFM_X)


    @pytest.mark.parametrize('_format', ('np', 'pd', 'pl', 'coo_array'))
    def test_rejects_no_samples(self, _shape, _kwargs, _X_np, _format):

        _CDT = CDT(**_kwargs)

        _X_base = np.empty((0, _shape[1]), dtype=np.float64)

        if _format == 'np':
            _X_wip = _X_base.copy()
        elif _format == 'pd':
            _X_wip = pd.DataFrame(_X_base)
        elif _format == 'pl':
            _X_wip = pl.from_numpy(_X_base)
        elif _format == 'coo_array':
            _X_wip = ss.coo_array(_X_base)
        else:
            raise Exception


        with pytest.raises(ValueError):
            _CDT.partial_fit(_X_wip)

        with pytest.raises(ValueError):
            _CDT.fit(_X_wip)

        with pytest.raises(ValueError):
            _CDT.fit_transform(_X_wip)

        _CDT.fit(_X_np)

        with pytest.raises(ValueError) as e:
            _CDT.transform(_X_wip)
        assert not isinstance(e.value, NotFittedError)

        with pytest.raises(ValueError) as e:
            _CDT.inverse_transform(_X_wip)
        assert not isinstance(e.value, NotFittedError)


    @pytest.mark.parametrize('_format', ('np', 'pd', 'pl', 'csc_array'))
    @pytest.mark.parametrize('_diff', ('more', 'less', 'same'))
    def test_rejects_bad_num_features(
        self, _X_factory, _shape, _dupls, _kwargs, _columns, _X_np,
        _format, _diff
    ):
        # ** ** ** **
        # THERE CANNOT BE "BAD NUM FEATURES" FOR fit & fit_transform
        # THE MECHANISM FOR partial_fit & transform IS DIFFERENT FROM inverse_transform
        # partial_fit & transform is handled by _check_n_features
        # inverse_transform has special code
        # ** ** ** **

        # SHOULD RAISE ValueError WHEN COLUMNS DO NOT EQUAL NUMBER OF
        # COLUMNS SEEN ON FIRST FIT

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
        _new_dupls_dict = {
            'same': _dupls,
            'less': [[0,_shape[1]-2], [1, _shape[1]-2]],
            'more': _dupls
        }

        _X_wip = _X_factory(
            _dupl=_new_dupls_dict[_diff],
            _has_nan=False, _format=_format, _dtype='flt',
            _columns=_columns_dict[_diff],
            _constants=None, _zeros=0,
            _shape=_new_shape_dict[_diff]
        )

        _CDT = CDT(**_kwargs)
        _CDT.fit(_X_np)

        if _diff == 'same':
            _CDT.partial_fit(_X_wip)
            _CDT.transform(_X_wip)
        else:
            with pytest.raises(ValueError) as e:
                _CDT.partial_fit(_X_wip)
            assert not isinstance(e.value, NotFittedError)
            with pytest.raises(ValueError) as e:
                _CDT.transform(_X_wip)
            assert not isinstance(e.value, NotFittedError)

    # END SHAPE #############################################################


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

        TestCls = CDT(**_kwargs)

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
            pybear_warn += ('NO_HDR_DF' in _objs and 'NO_HDR_DF' in _objs)
            # IF NONE OF THEM HAD A HEADER, THEN NO WARNING
            pybear_warn -= ('DF1' not in _objs and 'DF2' not in _objs)
            pybear_warn = max(0, pybear_warn)

        del _objs

        if pybear_exception:
            # this raises in _check_feature_names
            TestCls.partial_fit(fst_fit_X)
            with pytest.raises(ValueError) as e:
                TestCls.partial_fit(scd_fit_X)
                TestCls.transform(trfm_X)
            assert not isinstance(e.value, NotFittedError)
        elif pybear_warn:
            TestCls.partial_fit(fst_fit_X)
            with pytest.warns():
                TestCls.partial_fit(scd_fit_X)
                TestCls.transform(trfm_X)
        else:
            # SHOULD NOT EXCEPT OR WARN
            TestCls.partial_fit(fst_fit_X)
            TestCls.partial_fit(scd_fit_X)
            TestCls.transform(trfm_X)


@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestPartialFit:

    #     def partial_fit(
    #         self,
    #         X: DataContainer,
    #         y: any=None
    #     ) -> Self:


    @pytest.mark.parametrize('_format',
        ('np', 'pd', 'pl', 'csc_matrix', 'bsr_array', 'coo_matrix')
    )
    def test_X_is_not_mutated(
        self, _X_factory, _columns, _shape, _kwargs, _dupls, _format
    ):
        _X_wip = _X_factory(
            _dupl=_dupls, _has_nan=False, _format=_format, _dtype='flt',
            _columns=_columns, _constants=None, _noise=0, _zeros=None,
            _shape=_shape
        )

        if _format == 'np':
            assert _X_wip.flags['C_CONTIGUOUS'] is True

        try:
            _X_wip_before = _X_wip.copy()
        except:
            _X_wip_before = _X_wip.clone()


        # verify _X_wip does not mutate in partial_fit()
        _CDT = CDT(**_kwargs).partial_fit(_X_wip)


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
        (-1,0,1, np.pi, True, False, None, 'trash', [1,2], {1,2}, {'a':1},
        lambda x: x, min)
    )
    def test_fit_partial_fit_accept_Y_equals_anything(self, _kwargs, _X_np, _stuff):
        CDT(**_kwargs).partial_fit(_X_np, _stuff)
        CDT(**_kwargs).fit(_X_np, _stuff)


    def test_conditional_access_to_partial_fit_and_fit(self, _X_np, _kwargs):

        TestCls = CDT(**_kwargs)

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


    @pytest.mark.parametrize('_keep', ('first', 'last', 'random'))
    def test_many_partial_fits_equal_one_big_fit(
        self, _X_np, _kwargs, _shape, _keep
    ):

        # **** **** **** **** **** **** **** **** **** **** **** **** ****
        # THIS TEST IS CRITICAL FOR VERIFYING THAT transform PULLS THE
        # SAME COLUMN INDICES FOR ALL CALLS TO transform() WHEN
        # keep=='random'
        # **** **** **** **** **** **** **** **** **** **** **** **** ****

         # _X_np has no nans

        _kwargs['keep'] = _keep

        # ** ** ** ** ** ** ** ** ** ** **
        # TEST THAT ONE-SHOT partial_fit/transform == ONE-SHOT fit/transform
        OneShotPartialFitTestCls = CDT(**_kwargs).partial_fit(_X_np)

        OneShotFullFitTestCls = CDT(**_kwargs).fit(_X_np)

        _ = OneShotPartialFitTestCls.duplicates_
        __ = OneShotFullFitTestCls.duplicates_
        assert len(_) == len(__)
        for idx in range(len(_)):
            assert np.array_equal(_[idx], __[idx])
        del _, __

        ONE_SHOT_PARTIAL_FIT_TRFM_X = \
            OneShotPartialFitTestCls.transform(_X_np, copy=True)

        ONE_SHOT_FULL_FIT_TRFM_X = \
            OneShotFullFitTestCls.transform(_X_np, copy=True)

        # since keep=='random' can keep different column indices for
        # the different instances (OneShotPartialFitTestCls,
        # OneShotFullFitTestCls), it would probably be better to
        # avoid a mountain of complexity to prove out conditional
        # column equalities between the 2, just assert shape is same.
        assert ONE_SHOT_PARTIAL_FIT_TRFM_X.shape == \
               ONE_SHOT_FULL_FIT_TRFM_X.shape

        if _keep != 'random':
            assert np.array_equal(
                ONE_SHOT_PARTIAL_FIT_TRFM_X,
                ONE_SHOT_FULL_FIT_TRFM_X
            ), f"one shot partial fit trfm X != one shot full fit trfm X"

        del OneShotPartialFitTestCls, OneShotFullFitTestCls
        del ONE_SHOT_PARTIAL_FIT_TRFM_X, ONE_SHOT_FULL_FIT_TRFM_X

        # END TEST THAT ONE-SHOT partial_fit/transform==ONE-SHOT fit/transform
        # ** ** ** ** ** ** ** ** ** ** **

        # ** ** ** ** ** ** ** ** ** ** **
        # TEST PARTIAL FIT DUPLS ARE THE SAME WHEN FULL DATA IS partial_fit() 2X
        SingleFitTestClass = CDT(**_kwargs).fit(_X_np)
        _ = SingleFitTestClass.duplicates_

        DoublePartialFitTestClass = CDT(**_kwargs)
        DoublePartialFitTestClass.partial_fit(_X_np)
        __ = DoublePartialFitTestClass.duplicates_
        DoublePartialFitTestClass.partial_fit(_X_np)
        ___ = DoublePartialFitTestClass.duplicates_

        assert len(_) == len(__) == len(___)
        for idx in range(len(_)):
            assert np.array_equal(_[idx], __[idx])
            assert np.array_equal(_[idx], ___[idx])

        del _, __, ___, SingleFitTestClass, DoublePartialFitTestClass

        # END PARTIAL FIT DUPLS ARE THE SAME WHEN FULL DATA IS partial_fit() 2X
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
        ), f"agglomerated X chunks != original X"

        PartialFitTestCls = CDT(**_kwargs)
        OneShotFitTransformTestCls = CDT(**_kwargs)

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
            ), f"trfm X from partial fit / partial trfm != one-shot fit/trfm X"

            assert np.array_equiv(
                FULL_TRFM_X_ONE_SHOT_FIT_TRANSFORM.astype(str),
                FULL_TRFM_X_FROM_PARTIAL_FIT_ONESHOT_TRFM.astype(str)
            ), f"trfm X from partial fits / one-shot trfm != one-shot fit/trfm X"
        elif _keep == 'random':
            assert np.array_equiv(
                FULL_TRFM_X_FROM_PARTIAL_FIT_PARTIAL_TRFM.astype(str),
                FULL_TRFM_X_FROM_PARTIAL_FIT_ONESHOT_TRFM.astype(str)
            ), (f"trfm X from partial fit / partial trfm != "
                f"trfm X from partial fit / one-shot trfm")


    @pytest.mark.parametrize('_dtype', ('flt', 'int', 'obj', 'hybrid'))
    @pytest.mark.parametrize('_has_nan', (False, 5))
    def test_dupl_accuracy_over_many_partial_fits(
        self, _kwargs, _X_factory, _dtype, _has_nan
    ):

        # verify correct progression of reported duplicates as partial fits are done.
        # rig a set of arrays that have progressively decreasing duplicates

        _chunk_shape = (50, 20)  # must have at least 10 columns for dupls to work

        _start_dupl = [[0, 7], [2, 4, _chunk_shape[1] - 1], [3, 5, _chunk_shape[1] - 2]]

        _new_kwargs = deepcopy(_kwargs)
        _new_kwargs['equal_nan'] = True

        PartialFitTestCls = CDT(**_new_kwargs)

        # build a pool of non-dupls to fill the dupls in X along the way
        # build a starting data object for first partial fit, using full dupls
        # build a y vector
        # do a verification partial_fit, assert reported dupls for original X
        # make a holder for all the different _wip_Xs, to do one big fit at the end
        # for however many times u want to do this:
        #   randomly replace one of the dupls with non-dupl column
        #   partial_fit
        #   assert reported dupls - should be one less (the randomly chosen)
        # at the very end, stack all the _wip_Xs, do one big fit, verify dupls

        _pool_X = _X_factory(
            _dupl=None,  # <============
            _has_nan=_has_nan, _format='np', _dtype=_dtype,
            _columns=None, _zeros=None, _shape=_chunk_shape
        )

        _wip_X = _X_factory(
            _dupl=_start_dupl,  # <============
            _has_nan=_has_nan, _format='np', _dtype=_dtype,
            _columns=None, _zeros=None, _shape=_chunk_shape
        )

        y_np = np.random.randint(0, 2, (_chunk_shape[0]))

        # verify IM sees the dupl columns correctly ** * ** * ** *
        # this also sets the original dupl columns in PartialFitTestCls
        _dupl_columns = \
            PartialFitTestCls.partial_fit(_wip_X, y_np).duplicates_
        assert len(_dupl_columns) == len(_start_dupl)
        for idx in range(len(_start_dupl)):
            assert np.array_equal(
                _dupl_columns[idx],
                _start_dupl[idx]
            )
        del _dupl_columns
        # END verify IM sees the dupl columns correctly ** * ** * ** *

        # create a holder for the the original dupl column idxs
        _dupl_pool = list(itertools.chain(*_start_dupl))

        X_HOLDER = []
        X_HOLDER.append(_wip_X)

        # take out only half of the dupls (arbitrary) v^v^v^v^v^v^v^v^v^v^v^v^v
        for trial in range(len(_dupl_pool)//2):

            random_dupl = np.random.choice(_dupl_pool, 1, replace=False)[0]

            # take the random dupl of out _start_dupl and _dupl_pool,
            # and take a column out of the X pool to patch the dupl in _wip_X
            for _idx, _set in enumerate(reversed(_start_dupl)):
                try:
                    _start_dupl[_idx].remove(random_dupl)
                    if len(_start_dupl[_idx]) == 1:
                        # gotta take that dangling dupl out of dupl pool!
                        _dupl_pool.remove(_start_dupl[_idx][0])
                        # and out of _start_dupl by deleting the whole set
                        del _start_dupl[_idx]
                    break
                except:
                    continue
            else:
                raise Exception(f"could not find dupl idx in _start_dupl")

            _dupl_pool.remove(random_dupl)

            # now that random_dupl has been taken out of _start_dupl,
            # it may have been in the first position of a set which would
            # change the sorting of the sets. so re-sort the sets
            _start_dupl = sorted(_start_dupl, key=lambda x: x[0])

            # column from X is a doppleganger, column from pool shouldnt be
            # but verify anyway ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
            _from_X = _wip_X[:, random_dupl]
            _from_pool = _pool_X[:, random_dupl]
            assert not np.array_equal(
                _from_X[np.logical_not(nan_mask(_from_X))],
                _from_pool[np.logical_not(nan_mask(_from_pool))]
            )
            del _from_X, _from_pool
            # END verify ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

            _wip_X[:, random_dupl] = _pool_X[:, random_dupl].copy()

            X_HOLDER.append(_wip_X)

            # fit PartialFitTestCls on the new _wip_X
            # verify correctly reported dupls after this partial_fit
            _dupl_columns = \
                PartialFitTestCls.partial_fit(_wip_X, y_np).duplicates_
            assert len(_dupl_columns) == len(_start_dupl)
            for idx in range(len(_start_dupl)):
                assert np.array_equal(_dupl_columns[idx], _start_dupl[idx]), \
                    f"{_dupl_columns=}, {_start_dupl=}"

        # END take out only half of the dupls (arbitrary) v^v^v^v^v^v^v^v^v^v^v

        # we now have full X_HOLDER, which holds _wip_Xs with progressively
        # fewer duplicate columns
        # and PartialFitTestCls, which was fit sequentially on the _wip_Xs

        _partial_fit_dupl_columns = PartialFitTestCls.duplicates_
        # do a one-shot fit, compare results
        # stack all the _wip_Xs
        OneShotFitTestCls = CDT(**_new_kwargs).fit(np.vstack(X_HOLDER), y_np)
        _one_shot_dupl_columns = OneShotFitTestCls.duplicates_
        # remember that _start_dupls has dupl idxs popped out of it
        # as non-dupl columns were put into _wip_X
        assert len(_one_shot_dupl_columns) == len(_start_dupl)
        assert len(_partial_fit_dupl_columns) == len(_start_dupl)
        for idx, group in enumerate(_start_dupl):
            assert np.array_equal(_one_shot_dupl_columns[idx], group)
            assert np.array_equal(_partial_fit_dupl_columns[idx], group)

    # dont really need to test accuracy, see _partial_fit

# pizza pick up here

@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestTransform:

    #     def transform(
    #         self,
    #         X: DataContainer,
    #         *,
    #         copy: bool = None
    #     ) -> DataContainer:


    @pytest.mark.parametrize('_copy',
        (-1, 0, 1, 3.14, True, False, None, 'junk', [0, 1], (1,), {'a': 1}, min)
    )
    def test_copy_validation(self, _X_np, _shape, _kwargs, _copy):

        _CDT = CDT(**_kwargs)
        _CDT.fit(_X_np)

        if isinstance(_copy, (bool, type(None))):
            _CDT.transform(_X_np, copy=_copy)
        else:
            with pytest.raises(TypeError):
                _CDT.transform(_X_np, copy=_copy)


    @pytest.mark.parametrize('x_input_type', ('np', 'pd', 'pl', 'csc_array'))
    @pytest.mark.parametrize('output_type', [None, 'default', 'pandas', 'polars'])
    def test_output_types(
        self, _X_factory, _columns, _shape, _kwargs, _dupls, x_input_type,
        output_type
    ):

        # pizza this needs a reaming

        _X_wip = _X_factory(
            _dupl=_dupls,
            _has_nan=False,
            _format=x_input_type,
            _dtype='flt',
            _columns=None,
            _constants=None,
            _noise=1e-9,
            _zeros=None,
            _shape=_shape
        )

        TestCls = CDT(**_kwargs)
        TestCls.set_output(transform=output_type)

        TRFM_X = TestCls.fit_transform(_X_wip)

        # if output_type is None, should return same type as given
        if output_type is None:
            assert type(TRFM_X) == type(_X_wip), \
                (f"output_type is None, X output type ({type(TRFM_X)}) != "
                 f"X input type ({type(_X_wip)})")
        # if output_type 'default', should return np array no matter what given
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


    @pytest.mark.parametrize('_format',
        ('np', 'pd', 'pl', 'coo_matrix', 'dok_array', 'bsr_array')
    )
    def test_X_is_not_mutated(
            self, _X_factory, _columns, _shape, _kwargs, _dupls, _format
    ):
        _X_wip = _X_factory(
            _dupl=_dupls, _has_nan=False, _format=_format, _dtype='flt',
            _columns=_columns,
            _constants=None, _noise=0, _zeros=None, _shape=_shape
        )


        if _format == 'np':
            assert _X_wip.flags['C_CONTIGUOUS'] is True

        try:
            _X_wip_before = _X_wip.copy()
        except:
            _X_wip_before = _X_wip.clone()

        _CDT = CDT(**_kwargs).partial_fit(_X_wip)

        # verify _X_wip does not mutate in partial_fit()
        assert isinstance(_X_wip, type(_X_wip_before))
        assert _X_wip.shape == _X_wip_before.shape
        if isinstance(_X_wip, np.ndarray):
            assert _X_wip.flags['C_CONTIGUOUS'] is True

        if hasattr(_X_wip_before, 'toarray'):
            assert np.array_equal(
                _X_wip.toarray(),
                _X_wip_before.toarray()
            )
        elif isinstance(_X_wip_before, pd.core.frame.DataFrame):
            assert _X_wip.equals(_X_wip_before)
        else:
            assert np.array_equal(_X_wip_before, _X_wip)
        # END FROM PARTIAL_FIT

        # TRANSFORM
        _X_wip = _X_factory(
            _dupl=_dupls,
            _has_nan=False,
            _format=_format,
            _dtype='flt',
            _columns=_columns if _format in ['pd', 'pl'] else None,
            _constants=None,
            _zeros=0,
            _shape=_shape
        )

        try:
            _X_wip_before_transform = _X_wip.copy()
        except:
            _X_wip_before_transform = _X_wip.clone()

        _CDT = CDT(**_kwargs)
        _CDT.fit(_X_wip)

        out = _CDT.transform(_X_wip, copy=True)
        assert isinstance(out, type(_X_wip))

        # verify _X_wip does not mutate in transform() when copy=True
        # when copy=False anything goes
        assert isinstance(_X_wip, type(_X_wip_before_transform))
        assert _X_wip.shape == _X_wip_before_transform.shape

        if hasattr(_X_wip_before_transform, 'toarray'):
            assert np.array_equal(
                _X_wip.toarray(),
                _X_wip_before_transform.toarray()
            )
        elif isinstance(_X_wip_before_transform, pd.core.frame.DataFrame):
            assert _X_wip.equals(_X_wip_before_transform)
        else:
            assert np.array_equal(_X_wip_before_transform, _X_wip)
        # END TRANSFORM


    def test_one_all_nans(self, _X_factory, _kwargs, _shape):

        _X = _X_factory(
            _dupl=[[0,1]],
            _has_nan=False,
            _format='np',
            _dtype='flt',
            _columns=None,
            _zeros=None,
            _shape=(_shape[0], 3)
        )

        # set a column to all nans
        _X[:, -1] = np.nan
        # verify last column is all nans
        assert all(nan_mask_numerical(_X[:, -1]))

        # 2nd column should drop, should have 2 columns, last is all np.nan

        out = CDT(**_kwargs).fit_transform(_X)

        assert np.array_equal(out[:, 0], _X[:, 0])
        assert all(nan_mask_numerical(out[:, -1]))


    def test_two_all_nans(self, _X_factory, _kwargs, _shape):

        _X = _X_factory(
            _dupl=[[0,1]],
            _has_nan=False,
            _format='np',
            _dtype='flt',
            _columns=None,
            _zeros=None,
            _shape=(_shape[0], 4)
        )

        # set last 2 columns to all nans
        _X[:, [-2, -1]] = np.nan
        # verify last column is all nans
        assert all(nan_mask_numerical(_X[:, -1]))
        assert all(nan_mask_numerical(_X[:, -2]))

        # 2nd & 4th column should drop, should have 2 columns, last is all np.nan

        out = CDT(**_kwargs).fit_transform(_X)

        assert np.array_equal(out[:, 0], _X[:, 0])
        assert all(nan_mask_numerical(out[:, -1]))


    @pytest.mark.parametrize('same_or_diff', ('_same', '_diff'))
    @pytest.mark.parametrize('x_format', ('np', 'pd', 'pl', 'coo_array'))
    def test_all_columns_the_same_or_different(
            self, _X_factory, _kwargs, same_or_diff, x_format, _columns, _shape
    ):

        TEST_X = _X_factory(
            _dupl=[list(range(_shape[1]))] if same_or_diff == '_same' else None,
            _has_nan=False,
            _format=x_format,
            _dtype='flt',
            _shape=_shape
        )

        out = CDT(**_kwargs).fit_transform(TEST_X)

        if same_or_diff == '_same':
            assert out.shape[1] == 1
        elif same_or_diff == '_diff':
            assert out.shape[1] == _shape[1]


    # See CDTTransform_accuracy for accuracy tests


@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestFitTransform:


    @pytest.mark.parametrize('x_input_type', ('np', 'pd', 'pl', 'csc_array'))
    @pytest.mark.parametrize('output_type', [None, 'default', 'pandas', 'polars'])
    def test_output_types(
        self, _X_factory, _columns, _shape, _kwargs, _dupls, x_input_type,
        output_type
    ):

        # pizza this needs a reaming

        _X_wip = _X_factory(
            _dupl=_dupls,
            _has_nan=False,
            _format=x_input_type,
            _dtype='flt',
            _columns=None,
            _constants=None,
            _noise=1e-9,
            _zeros=None,
            _shape=_shape
        )

        TestCls = CDT(**_kwargs)
        TestCls.set_output(transform=output_type)

        TRFM_X = TestCls.fit_transform(_X_wip)

        # if output_type is None, should return same type as given
        if output_type is None:
            assert type(TRFM_X) == type(_X_wip), \
                (f"output_type is None, X output type ({type(TRFM_X)}) != "
                 f"X input type ({type(_X_wip)})")
        # if output_type 'default', should return np array no matter what given
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



@pytest.mark.skipif(bypass is True, reason=f"bypass")
class TestInverseTransform:

    #     def inverse_transform(
    #         self,
    #         X: DataContainer,
    #         *,
    #         copy: bool = None
    #     ) -> DataContainer:


    # - num columns must equal num columns in column_mask_


    @pytest.mark.parametrize('_copy',
        (-1, 0, 1, 3.14, True, False, None, 'junk', [0,1], (1,), {'a': 1}, min)
    )
    def test_copy_validation(self, _X_np, _shape, _kwargs, _copy):

        _CDT = CDT(**_kwargs)
        _CDT.fit(_X_np)

        if isinstance(_copy, (bool, type(None))):
            _CDT.inverse_transform(_X_np[:, _CDT.column_mask_], copy=_copy)
        else:
            with pytest.raises(TypeError):
                _CDT.inverse_transform(_X_np[:, _CDT.column_mask_], copy=_copy)



    @pytest.mark.parametrize('_format',
        (
            'np', 'pd', 'pl', 'csr_matrix', 'csc_matrix', 'coo_matrix', 'dia_matrix',
            'lil_matrix', 'dok_matrix', 'bsr_matrix', 'csr_array', 'csc_array',
            'coo_array', 'dia_array', 'lil_array', 'dok_array', 'bsr_array'
        )
    )
    @pytest.mark.parametrize('_dtype', ('int', 'flt', 'str', 'obj', 'hybrid'))
    @pytest.mark.parametrize('_has_nan', (True, False))
    @pytest.mark.parametrize('_keep', ('first', 'last', 'random'))
    @pytest.mark.parametrize('_copy', (True, False))
    def test_accuracy(
        self, _X_factory, _columns, _kwargs, _shape, _format, _dtype,
        _has_nan, _keep, _dupls, _copy
    ):

        # may not need to test accuracy here, see _inverse_transform,
        # but it is pretty straightforward. affirms the CDT class
        # inverse_transform method works correctly, above and beyond just
        # the _inverse_transform function called within.

        # set_output does not control the output container for inverse_transform
        # the output container is always the same as passed

        # skip impossible conditions -- -- -- -- -- -- -- -- -- -- -- -- -- --
        if _format not in ('np', 'pd', 'pl') and _dtype not in ('int', 'flt'):
            pytest.skip(reason=f"scipy sparse cant take non-numeric")
        # END skip impossible conditions -- -- -- -- -- -- -- -- -- -- -- -- --

        # build X ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        _X_wip = _X_factory(
            _dupl=_dupls,
            _has_nan=_has_nan,
            _format=_format,
            _dtype=_dtype,
            _columns=_columns,
            _constants=None,
            _zeros=0,
            _shape=_shape
        )

        # funky pd nans are making equality tests difficult
        # pizza see if u can fix the root cause. IM doesnt need this.
        # if _format == 'pd':
        #     _X_wip[nan_mask(_X_wip)] = np.nan
        # if _format == 'pl':
        #     _X_wip[nan_mask(_X_wip)] = None

        if _format == 'np':
            _base_X = _X_wip.copy()
        elif _format in ['pd', 'pl']:
            _base_X = _X_wip.to_numpy()
        elif hasattr(_X_wip, 'toarray'):
            _base_X = _X_wip.toarray()
        else:
            raise Exception
        # END build X ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

        try:
            _X_wip_before_inv_tr = _X_wip.copy()
        except:
            _X_wip_before_inv_tr = _X_wip.clone()

        _kwargs['keep'] = _keep

        _CDT = CDT(**_kwargs)

        # fit v v v v v v v v v v v v v v v v v v v v
        _CDT.fit(_X_wip)
        # fit ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^

        # transform v v v v v v v v v v v v v v v v v v
        TRFM_X = _CDT.transform(_X_wip, copy=True)
        assert isinstance(TRFM_X, type(_X_wip))
        # transform ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^

        # inverse transform v v v v v v v v v v v v v v v
        INV_TRFM_X = _CDT.inverse_transform(
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
        assert INV_TRFM_X.shape[1] == _CDT.n_features_in_, \
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
            # THE ORDER IS IMPORTANT HERE. CONVERT TO PANDAS FIRST.
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

        if isinstance(NP_INV_TRFM_X, pd.core.frame.DataFrame):
            assert np.array_equal(NP_INV_TRFM_X.columns, _columns)

        # manage equal_nan for num or str
        try:
            NP_INV_TRFM_X.astype(np.float64)
            # when num
            assert np.array_equal(NP_INV_TRFM_X, _base_X, equal_nan=True), \
                f"inverse transform of transformed data does not equal original"

            assert np.array_equal(
                NP_TRFM_X,
                NP_INV_TRFM_X[:, _CDT.column_mask_],
                equal_nan=True
            ), \
                (f"output of inverse_transform() does not reduce back to the "
                 f"output of transform()")
        except:
            # when str
            # CDT converts all nan-likes to np.nan, need to standardize them
            # in the inputs here for array_equal against the outputs
            # for string data, array_equal does not accept equal_nan param
            _base_X[nan_mask_string(_base_X)] = 'nan'
            NP_TRFM_X[nan_mask_string(NP_TRFM_X)] = 'nan'
            NP_INV_TRFM_X[nan_mask_string(NP_INV_TRFM_X)] = 'nan'

            assert np.array_equal(NP_INV_TRFM_X, _base_X), \
                f"inverse transform of transformed data != original data"

            assert np.array_equal(
                NP_TRFM_X, NP_INV_TRFM_X[:, _CDT.column_mask_]
            ), \
                (f"output of inverse_transform() does not reduce back to the "
                 f"output of transform()")


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
            elif hasattr(_X_wip_before_inv_tr, 'columns'):
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
        self, _X_factory, _columns, _shape, _kwargs, _dupls, _format
    ):

        # set_output does not control the output container for inverse_transform
        # the output container is always the same as passed

        _X_wip = _X_factory(
            _dupl=_dupls,
            _has_nan=False,
            _format=_format,
            _dtype='flt',
            _columns=_columns,
            _constants=None,
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
        _CDT = CDT(**_kwargs)
        _CDT.fit(_X_wip)
        TRFM_X = _CDT.transform(_X_wip, copy=True)
        assert isinstance(TRFM_X, type(_X_wip))
        assert TRFM_X.shape[1] < _X_wip.shape[1]
        INV_TRFM_X = _CDT.inverse_transform(X=TRFM_X, copy=True)


        # ASSERTIONS v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v
        # output container is same as passed
        assert isinstance(_X_wip, type(_X_wip_before))
        assert _X_wip.shape == _X_wip_before.shape

        if isinstance(_X_wip_before, np.ndarray):
            assert np.array_equal(
                _X_wip_before, _X_wip, equal_nan=True
            )
            assert _X_wip.flags == _X_wip_before.flags
        elif hasattr(_X_wip_before, 'columns'):
            assert _X_wip.equals(_X_wip_before)
        elif hasattr(_X_wip_before, 'toarray'):
            assert np.array_equal(
                _X_wip.toarray(), _X_wip_before.toarray(), equal_nan=True
            )
        else:
            raise Exception


    @pytest.mark.parametrize('_format', ('np', 'pd', 'pl'))
    @pytest.mark.parametrize('_diff', ('more', 'less', 'same'))
    def test_rejects_bad_num_features(
        self, _X_np, _kwargs, _columns, _format, _diff
    ):

        # ** ** ** **
        # THERE CANNOT BE "BAD NUM FEATURES" FOR fit & fit_transform
        # THE MECHANISM FOR partial_fit & transform IS DIFFERENT FROM inverse_transform
        # ** ** ** **

        # RAISE ValueError WHEN COLUMNS DO NOT EQUAL NUMBER OF
        # COLUMNS RETAINED BY column_mask_

        _CDT = CDT(**_kwargs)
        _CDT.fit(_X_np)

        # build TRFM_X ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
        TRFM_X = _CDT.transform(_X_np)
        TRFM_MASK = _CDT.column_mask_
        if _diff == 'same':
            if _format == 'pd':
                TRFM_X = pd.DataFrame(data=TRFM_X, columns=_columns[TRFM_MASK])
            elif _format == 'pl':
                TRFM_X = pl.DataFrame(data=TRFM_X, schema=list(_columns[TRFM_MASK]))
        elif _diff == 'less':
            TRFM_X = TRFM_X[:, :-1]
            if _format == 'pd':
                TRFM_X = pd.DataFrame(data=TRFM_X, columns=_columns[TRFM_MASK][:-1])
            elif _format == 'pl':
                TRFM_X = pl.from_numpy(
                    data=TRFM_X, schema=list(_columns[TRFM_MASK][:-1])
                )
        elif _diff == 'more':
            TRFM_X = np.hstack((TRFM_X, TRFM_X))
            _COLUMNS = np.hstack((
                _columns[TRFM_MASK], np.char.upper(_columns[TRFM_MASK])
            ))
            if _format == 'pd':
                TRFM_X = pd.DataFrame(data=TRFM_X, columns=_COLUMNS)
            elif _format == 'pl':
                TRFM_X = pl.from_numpy(data=TRFM_X, schema=list(_COLUMNS))
        else:
            raise Exception
        # END build TRFM_X ** * ** * ** * ** * ** * ** * ** * ** * ** * **

        # Test the inverse_transform operation ** ** ** ** ** ** **
        if _diff == 'same':
            _CDT.inverse_transform(TRFM_X)
        else:
            with pytest.raises(ValueError):
                _CDT.inverse_transform(TRFM_X)
        # END Test the inverse_transform operation ** ** ** ** ** ** **

        del _CDT, TRFM_X, TRFM_MASK





