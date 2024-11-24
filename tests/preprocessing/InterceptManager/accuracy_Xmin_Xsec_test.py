# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.InterceptManager.InterceptManager import \
    InterceptManager as IM

from pybear.preprocessing.InterceptManager._partial_fit. \
    _parallel_constant_finder import _parallel_constant_finder

from pybear.preprocessing.ColumnDeduplicateTransformer._partial_fit. \
    _parallel_column_comparer import _parallel_column_comparer

from pybear.utilities import nan_mask

from copy import deepcopy
import numpy as np
import pandas as pd


import pytest






class TestAccuracy:


    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (20, 10)


    @staticmethod
    @pytest.fixture(scope='module')
    def _columns(_master_columns, _shape):
        return _master_columns.copy()[:_shape[1]]


    @staticmethod
    @pytest.fixture(scope='function')
    def _kwargs():
        return {
            'keep': 'first',
            'equal_nan': False,
            'rtol': 1e-5,
            'atol': 1e-8,
            'n_jobs': -1     # leave at -1, pizza set this after benchmarking
        }



    @pytest.mark.parametrize('X_format', ('np', 'pd', 'csr', 'csr', 'coo'))
    @pytest.mark.parametrize('X_dtype', ('flt', 'int', 'str', 'obj', 'hybrid'))
    @pytest.mark.parametrize('has_nan', (True, False))
    @pytest.mark.parametrize('constants', ('constants1', 'constants2', 'constants3'))
    @pytest.mark.parametrize('keep', ('first', 'last', 'random', 'none', 'int', 'string', 'callable', {'Intercept': 1}))
    @pytest.mark.parametrize('equal_nan', (True, False))
    def test_accuracy(self, _X_factory, _kwargs, X_format, X_dtype, has_nan,
        constants, keep, _columns, equal_nan, _shape
    ):

        # validate the test parameters
        assert keep in ['first', 'last', 'random', 'none'] or isinstance(keep, (int, dict, str)) or callable(keep)
        assert isinstance(equal_nan, bool)
        # END validate the test parameters

        # skip impossible combinations v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
        if X_dtype in ['str', 'obj', 'hybrid'] and X_format not in ['np', 'pd']:
            pytest.skip(reason=f"scipy sparse cant take str")

        if X_format == 'np' and X_dtype == 'int' and has_nan:
            pytest.skip(reason=f"numpy int dtype cant take 'nan'")
        # END skip impossible combinations v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^

        # set constants v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v
        if constants == 'constants1':
            constants = None
        elif constants == 'constants2':
            if X_dtype in ('flt', 'int'):
                constants = {0: 1, 2: 1, 9: 1}
            elif X_dtype in ('str', 'obj', 'hybrid'):
                constants = {0: 1, 2: 'a', 9: 'b'}
        elif constants == 'constants3':
            if X_dtype in ('flt', 'int'):
                constants = {0: 1, 1: 1, 6: np.nan, 8: 1}
            elif X_dtype in ('str', 'obj', 'hybrid'):
                constants = {0: 'a', 1: 'b', 6: 'nan', 8: '1'}
        else:
            raise Exception
        # END set constants v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v

        X = _X_factory(
            _dupl=None,
            _has_nan=has_nan,
            _format=X_format,
            _dtype=X_dtype,
            _columns=_columns,
            _constants=constants,
            _noise=1e-9,
            _zeros=None,
            _shape=_shape
        )

        # set keep v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
        if keep == 'string':
            keep = _columns[0]
        elif keep == 'int':
            if constants:
                keep = sorted(list(constants.keys()))[-1]
            else:
                keep = 0
        elif keep == 'callable':
            if constants:
                keep = lambda x: sorted(list(constants.keys()))[-1]
            else:
                keep = lambda x: 0
        else:
            # keep is not changed
            pass
        # END set keep v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^

        # set _kwargs
        _kwargs['keep'] = keep
        _kwargs['equal_nan'] = equal_nan

        TestCls = IM(**_kwargs)


        # retain original format
        _og_format = type(X)
        if isinstance(X, pd.core.frame.DataFrame):
            # need to adjust the dtypes if keep is dict
            if isinstance(keep, dict):
                from uuid import uuid4
                _key = list(keep.keys())[0]
                _value = keep[_key]
                _vector = pd.DataFrame({uuid4(): np.full(X.shape[0], _value)})
                try:
                    float(_value)
                    _dtype = np.float64
                except:
                    _dtype = object
                del _key, _value
                _og_dtype = pd.concat((X, _vector.astype(_dtype)), axis=1).dtypes
                del _vector
            else:
                _og_dtype = X.dtypes
        else:
            _og_dtype = X.dtype

        # manage constants
        exp_constants = deepcopy(constants or {})
        if has_nan and not equal_nan:
            exp_constants = {}
        if not equal_nan and len(exp_constants) and any(nan_mask(list(exp_constants.values()))):
            exp_constants = {k:v for k,v in exp_constants.items() if str(v) != 'nan'}

        # if data is not pd and user put in keep as feature_str, will raise
        raise_for_no_header_str_keep = False
        if X_format != 'pd' and isinstance(keep, str) and keep not in ('first', 'last', 'random', 'none'):
            raise_for_no_header_str_keep += 1

        # if data has no constants and user put in keep feature_str/int/callable, will raise
        raise_for_keep_non_constant = False
        has_constants = len(exp_constants) and not (has_nan and not equal_nan)
        if not has_constants:
            if callable(keep):
                raise_for_keep_non_constant += 1
            if isinstance(keep, int):
                raise_for_keep_non_constant += 1
            if isinstance(keep, str) and keep not in ('first', 'last', 'random', 'none'):
                raise_for_keep_non_constant += 1


        if raise_for_no_header_str_keep or raise_for_keep_non_constant:
            with pytest.raises(ValueError):
                TestCls = TestCls.fit(X)
            pytest.skip(reason=f"cant do anymore tests without fit")
        else:
            TestCls = TestCls.fit(X)

        del raise_for_keep_non_constant, raise_for_no_header_str_keep

        TRFM_X = TestCls.transform(X)

        if isinstance(TRFM_X, np.ndarray):
            assert TRFM_X.flags['C_CONTIGUOUS'] is True


        # ASSERTIONS ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

        # attributes:
        #     'n_features_in_'
        #     'feature_names_in_'
        #     'constant_columns_'
        #     'kept_columns_'
        #     'removed_columns_'
        #     'column_mask_'
        # and 'get_feature_names_out'

        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # returned format is same as given format
        assert isinstance(TRFM_X, _og_format)

        # returned dtypes are same as given dtypes
        if isinstance(TRFM_X, pd.core.frame.DataFrame):
            MASK = TestCls.column_mask_
            if isinstance(keep, dict):
                MASK = np.hstack((MASK, np.array([True], dtype=bool)))

            assert np.array_equal(TRFM_X.dtypes, _og_dtype[MASK])
            del MASK
        else:
            pass
            # pizza U dtypes are changing. come back and try to understand this
            # assert TRFM_X.dtype == _og_dtype
        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # attr 'n_features_in_' is correct
        assert TestCls.n_features_in_ == X.shape[1]
        assert len(TestCls.column_mask_) == TestCls.n_features_in_

        # attr 'feature_names_in_' is correct
        if X_dtype == 'pd':
            assert np.array_equal(TestCls.feature_names_in_, _columns)
            assert TestCls.feature_names_in_.dtype == object

        # number of columns in output is adjusted correctly for num constants
        exp_num_kept = _shape[1]
        if not has_nan or (has_nan and equal_nan):
            if len(exp_constants):
                exp_num_kept -= len(exp_constants)
                exp_num_kept +=  (keep != 'none') # this catches dict
            else:
                exp_num_kept += isinstance(keep, dict)
        elif has_nan and not equal_nan:
            exp_num_kept += isinstance(keep, dict)
        else:
            raise Exception(f"algorithm failure")

        # number of columns in output == number of columns in column_mask_
        assert TRFM_X.shape[1] == exp_num_kept
        assert sum(TestCls.column_mask_) + isinstance(keep, dict) == exp_num_kept

        # attr 'constant_columns_' is correct
        act_constants = TestCls.constant_columns_
        assert len(act_constants) == len(exp_constants)
        assert np.array_equal(list(act_constants.keys()), list(exp_constants.keys()))
        for idx, value in exp_constants.items():
            if str(value) == 'nan':
                assert str(value) == str(act_constants[idx])
            else:
                try:
                    float(value)
                    is_num = True
                except:
                    is_num = False

                if is_num:
                    assert np.isclose(float(value), float(act_constants[idx]), rtol=1e-5, atol=1e-8)
                else:
                    assert value == act_constants[idx]


        # keep ('first','last','random','none') is correct
        # also verify 'column_mask_' 'kept_columns_', 'removed_columns_'
        # 'get_feature_names_out_' has its own test module
        ref_column_mask = [True for _ in range(X.shape[1])]
        ref_kept_columns = {}
        ref_removed_columns = {}
        ref_feature_names_out = list(deepcopy(_columns))

        _sorted_constant_idxs = sorted(list(exp_constants.keys()))

        if keep == 'first':
            if len(_sorted_constant_idxs):
                _first = _sorted_constant_idxs[0]
                ref_kept_columns[_first] = exp_constants[_first]
                del _first

                for idx in _sorted_constant_idxs[1:]:
                    ref_column_mask[idx] = False
                    ref_removed_columns[idx] = exp_constants[idx]
                    ref_feature_names_out[idx] = None
        elif keep == 'last':
            if len(_sorted_constant_idxs):
                _last = _sorted_constant_idxs[-1]
                ref_kept_columns[_last] = exp_constants[_last]
                del _last

                for idx in _sorted_constant_idxs[:-1]:
                    ref_column_mask[idx] = False
                    ref_removed_columns[idx] = exp_constants[idx]
                    ref_feature_names_out[idx] = None
        elif keep == 'random':
            # cop out a little here, since we cant know what index
            # was kept, use TestCls attr for help
            # (even tho the attrs are something we are trying
            # to verify)
            if len(_sorted_constant_idxs):
                assert len(TestCls.kept_columns_) == 1
                _kept_idx = list(TestCls.kept_columns_.keys())[0]
                assert _kept_idx in exp_constants
                ref_kept_columns[_kept_idx] = exp_constants[_kept_idx]
                assert _kept_idx not in TestCls.removed_columns_

                for idx in _sorted_constant_idxs:
                    if idx == _kept_idx:
                        continue
                    ref_column_mask[idx] = False
                    ref_removed_columns[idx] = exp_constants[idx]
                    ref_feature_names_out[idx] = None
                del _kept_idx

        elif keep == 'none':
            # ref_kept_columns does not change
            for idx in _sorted_constant_idxs:
                ref_column_mask[idx] = False
                ref_removed_columns[idx] = exp_constants[idx]
                ref_feature_names_out[idx] = None
        elif isinstance(keep, dict):
            # ref_kept_columns does not change
            for idx in _sorted_constant_idxs:
                ref_column_mask[idx] = False
                ref_removed_columns[idx] = exp_constants[idx]
                ref_feature_names_out[idx] = None
            ref_feature_names_out.append(list(keep.keys())[0])
        elif isinstance(keep, int):
            # if no constants, should have excepted and skipped above
            ref_kept_columns[keep] = exp_constants[keep]
            for idx in _sorted_constant_idxs:
                if idx == keep:
                    continue
                else:
                    ref_column_mask[idx] = False
                    ref_removed_columns[idx] = exp_constants[idx]
                    ref_feature_names_out[idx] = None
        elif isinstance(keep, str):  # must be a feature str
            # if no constants, should have excepted and skipped above
            # if no header, should have excepted and skipped above
            _kept_idx = np.arange(len(_columns))[_columns == keep][0]
            ref_kept_columns[_kept_idx] = exp_constants[_kept_idx]
            for idx in _sorted_constant_idxs:
                if idx == _kept_idx:
                    continue
                else:
                    ref_column_mask[idx] = False
                    ref_removed_columns[idx] = exp_constants[idx]
                    ref_feature_names_out[idx] = None
            del _kept_idx
        elif callable(keep):
            # if no constants, should have excepted and skipped above
            _kept_idx = keep(X)
            ref_kept_columns[_kept_idx] = exp_constants[_kept_idx]
            for idx in _sorted_constant_idxs:
                if idx == _kept_idx:
                    continue
                else:
                    ref_column_mask[idx] = False
                    ref_removed_columns[idx] = exp_constants[idx]
                    ref_feature_names_out[idx] = None
        else:
            raise Exception


        # for convenient index management, positions to be dropped from
        # ref_feature_names_out were set to None, take those out now
        ref_feature_names_out = [_ for _ in ref_feature_names_out if _ is not None]

        # validate TestCls attrs against ref objects v^v^v^v^v^v^v^v^v^v^v^v^v^
        # column_mask_ ------------
        assert (sum(ref_column_mask) + isinstance(keep, dict)) == exp_num_kept
        assert np.array_equal(ref_column_mask, TestCls.column_mask_)
        # assert all columns that werent constants are in the output
        for col_idx in range(_shape[1]):
            if col_idx not in _sorted_constant_idxs:
                # pizza when this thing runs come back see if can get this to py True
                assert TestCls.column_mask_[col_idx] is np.True_



        # kept_columns_ -----------------
        _, __ = TestCls.kept_columns_, ref_kept_columns
        assert np.array_equal(
            sorted(list(_.keys())),
            sorted(list(__.keys()))
        )

        try:
            np.array(_).astype(np.float64)
            np.array(__).astype(np.float64)
            is_num = True
        except:
            is_num = False

        if is_num:
            assert np.allclose(
                list(_.values()),
                list(__.values()),
                equal_nan=True
            )
        else:
            _ = list(_.values())
            __ = list(__.values())
            assert len(_) == len(__)

            for idx in range(len(_)):
                try:
                    float(_[idx])
                    float(__[idx])
                    if str(_[idx]) == 'nan' or str(__[idx]) == 'nan':
                        raise Exception
                    is_num = True
                except:
                    is_num = False

                if is_num:
                    assert np.isclose(float(_[idx]), float(__[idx]))
                else:
                    assert str(_[idx]) == str(__[idx])

        del _, __
        # END kept_columns_ -----------------

        # removed_columns_ ------------------
        _, __ = TestCls.removed_columns_, ref_removed_columns
        assert np.array_equal(
            sorted(list(_.keys())),
            sorted(list(__.keys()))
        )

        try:
            np.array(_).astype(np.float64)
            np.array(__).astype(np.float64)
            is_num = True
        except:
            is_num = False

        if is_num:
            assert np.allclose(
                list(_.values()),
                list(__.values()),
                equal_nan=True
            )
        else:
            _ = list(_.values())
            __ = list(__.values())
            assert len(_) == len(__)

            for idx in range(len(_)):
                try:
                    float(_[idx])
                    float(__[idx])
                    if str(_[idx]) == 'nan' or str(__[idx]) == 'nan':
                        raise Exception
                    is_num = True
                except:
                    is_num = False

                if is_num:
                    assert np.isclose(float(_[idx]), float(__[idx]))
                else:
                    assert str(_[idx]) == str(__[idx])

        del _, __
        # END removed_columns_ ------------------
        # END validate TestCls attrs against ref objects v^v^v^v^v^v^v^v^v^v^v^


        # for retained columns, assert they are equal to themselves in
        # the original data. if the column is not retained, then assert
        # the columns in the original data is a constant
        _new_idx = -1
        _kept_idxs = np.arange(len(TestCls.column_mask_))[TestCls.column_mask_]

        if len(_sorted_constant_idxs):
            if keep == 'first':
                assert _sorted_constant_idxs[0] in _kept_idxs
            elif keep == 'last':
                assert _sorted_constant_idxs[-1] in _kept_idxs
            elif keep == 'random':
                assert sum([i in _kept_idxs for i in _sorted_constant_idxs]) == 1
            elif isinstance(keep, dict) or keep == 'none':
                assert sum([i in _kept_idxs for i in _sorted_constant_idxs]) == 0
        else:
            assert len(_kept_idxs) == X.shape[1]


        for _idx in range(_shape[1]):

            if _idx in _kept_idxs:
                _new_idx += 1

            if isinstance(X, np.ndarray):
                _out_col = TRFM_X[:, _new_idx]
                _og_col = X[:, _idx]
            elif isinstance(X, pd.core.frame.DataFrame):
                _out_col = TRFM_X.iloc[:, _new_idx].to_numpy()
                _og_col = X.iloc[:, _idx].to_numpy()
            else:
                _out_col = TRFM_X.tocsc()[:, [_new_idx]].toarray()
                _og_col = X.tocsc()[:, [_idx]].toarray()

            try:
                _out_col = _out_col.astype(np.float64)
                _og_col = _og_col.astype(np.float64)
            except:
                pass

            if _idx in _kept_idxs:

                assert _parallel_column_comparer(
                    _out_col,
                    _og_col,
                    _rtol=1e-5,
                    _atol=1e-8,
                    _equal_nan=True
                )
            else:
                assert _parallel_constant_finder(
                    _column=_og_col,
                    _equal_nan=equal_nan,
                    _rtol=1e-5,
                    _atol=1e-8
                )


        # END ASSERTIONS ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *









