# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.SlimPolyFeatures.SlimPolyFeatures import \
    SlimPolyFeatures as SlimPoly

from pybear.preprocessing.SlimPolyFeatures._partial_fit. \
    _parallel_constant_finder import _parallel_constant_finder

from pybear.preprocessing.SlimPolyFeatures._partial_fit. \
    _parallel_column_comparer import _parallel_column_comparer



from copy import deepcopy
import numpy as np
import pandas as pd

import pytest







pytest.skip(reason=f"pizza not finished", allow_module_level=True)


# pizza dont forget accuracy when X is 1 column


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
            'degree': 2,  # will be overwrit during test
            'min_degree': 1,  # will be overwrit during test
            'keep': 'first',
            'interaction_only': False,  # will be overwrit during test
            'scan_X': False,
            'sparse_output': False,
            'feature_name_combiner': 'as_indices',
            'rtol': 1e-5,
            'atol': 1e-8,
            'equal_nan': True,
            'n_jobs': 1
        }


    @pytest.mark.parametrize('X_format', ('np', 'pd'))
    @pytest.mark.parametrize('X_dtype', ('flt', 'int'))
    @pytest.mark.parametrize('degree', (2, 3))
    @pytest.mark.parametrize('min_degree', (1, 2))
    @pytest.mark.parametrize('interaction_only', (True, False))
    def test_accuracy(
        self, _X_factory, _kwargs, X_format, X_dtype, degree, min_degree,
        interaction_only, _columns, _shape
    ):

        X = _X_factory(
            _dupl=None,
            _has_nan=False,
            _format=X_format,
            _dtype=X_dtype,
            _columns=_columns,
            _shape=_shape
        )

        # set _kwargs
        _kwargs['X_format'] = X_format
        _kwargs['X_dtype'] = X_dtype
        _kwargs['degree'] = degree
        _kwargs['min_degree'] = min_degree
        _kwargs['interaction_only'] = interaction_only


        TestCls = SlimPoly(**_kwargs)


        # retain original format
        _og_format = type(X)

        TestCls.fit(X)

        TRFM_X = TestCls.transform(X)


        # ASSERTIONS ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

        # attributes:
        #     'n_features_in_'
        #     'feature_names_in_'
        #     'constant_columns_'
        #     'kept_columns_'
        #     'removed_columns_'
        #     'column_mask_'

        if isinstance(TRFM_X, np.ndarray):
            assert TRFM_X.flags['C_CONTIGUOUS'] is True

        # returned format is same as given format
        assert isinstance(TRFM_X, _og_format)

        # returned dtypes are same as given dtypes ** * ** * ** * ** * **
        import os
        if isinstance(TRFM_X, pd.core.frame.DataFrame):
            MASK = TestCls.column_mask_
            if isinstance(keep, dict):
                # remember that above we stacked a fudged intercept column
                # to the df to get all the dtypes in one shot. so now here
                # we need to adjust column_mask_ to get the extra column
                MASK = np.hstack((MASK, np.array([True], dtype=bool)))

            # dtypes could be shape[1] or (shape[1] + isinstance(keep, dict))
            assert np.array_equal(TRFM_X.dtypes, _og_dtype[MASK])
            del MASK
        elif '<U' in str(_og_dtype):
            # str dtypes are changing in _transform() at
            # _X = np.hstack((
            #     _X,
            #     np.full((_X.shape[0], 1), _value)
            # ))
            # there does not seem to be an obvious connection between what
            # the dtype of _value is and the resultant dtype (for example,
            # _X with dtype '<U10' when appending float(1.0), the output dtype
            # is '<U21' (???, maybe floating point error on the float?) )
            assert '<U' in str(TRFM_X.dtype)
        elif os.name == 'nt' and 'int' in str(_og_dtype).lower():
            # on windows (verified not macos or linux), int dtypes are
            # changing to int64, in _transform() at
            # _X = np.hstack((
            #     _X,
            #     np.full((_X.shape[0], 1), _value)
            # ))
            assert 'int' in str(TRFM_X.dtype).lower()
        else:
            assert TRFM_X.dtype == _og_dtype
        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # attr 'n_features_in_' is correct
        assert TestCls.n_features_in_ == X.shape[1]

        # attr 'feature_names_in_' is correct
        if X_format == 'pd':
            assert np.array_equal(TestCls.feature_names_in_, _columns)
            assert TestCls.feature_names_in_.dtype == object
        else:
            assert not hasattr (TestCls, 'feature_names_in_')

        # number of columns in output is adjusted correctly for num constants
        exp_num_kept = _shape[1]  # start with full num of columns
        if not has_nan or (has_nan and equal_nan):
            if len(exp_constants):
                # if there are constants, 'none' removes all n of them,
                # but all other 'keep' arguments remove n-1
                exp_num_kept -= len(exp_constants)
                exp_num_kept +=  (keep != 'none') # this catches dict
            else:
                # if there are no constants, all that could happen is that
                # keep dict is appended
                exp_num_kept += isinstance(keep, dict)
        elif has_nan and not equal_nan:
            # in this case there are no constant columns, all that could
            # happen is that keep dict is appended
            exp_num_kept += isinstance(keep, dict)
        else:
            raise Exception(f"algorithm failure")


        # number of columns in output == number of expected
        assert TRFM_X.shape[1] == exp_num_kept


        # attr 'constant_columns_' is correct ------------------------------
        act_constants = TestCls.constant_columns_
        assert len(act_constants) == len(exp_constants)
        assert np.array_equal(
            list(act_constants.keys()),
            list(exp_constants.keys())
        )
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
                    assert np.isclose(
                        float(value),
                        float(act_constants[idx]),
                        rtol=1e-5,
                        atol=1e-8
                    )
                else:
                    assert value == act_constants[idx]
        # END attr 'constant_columns_' is correct ----------------------------

        # keep ('first','last','random','none') is correct
        # also verify 'column_mask_' 'kept_columns_', 'removed_columns_'
        ref_column_mask = [True for _ in range(X.shape[1])]
        ref_kept_columns = {}
        ref_removed_columns = {}
        if X_format == 'pd':
            ref_feature_names_out = list(deepcopy(_columns))
        else:
            ref_feature_names_out = [f'x{i}' for i in range(_shape[1])]

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
            # if there are no constants, then ref objects do not change
        elif keep == 'last':
            if len(_sorted_constant_idxs):
                _last = _sorted_constant_idxs[-1]
                ref_kept_columns[_last] = exp_constants[_last]
                del _last

                for idx in _sorted_constant_idxs[:-1]:
                    ref_column_mask[idx] = False
                    ref_removed_columns[idx] = exp_constants[idx]
                    ref_feature_names_out[idx] = None
            # if there are no constants, then ref objects do not change
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
            # if there are no constants, then ref objects do not change
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

        # feature_names_out ------------
        assert np.array_equal(
            TestCls.get_feature_names_out(),
            ref_feature_names_out
        )

        del ref_feature_names_out
        # END feature_names_out ------------


        # column_mask_ ------------
        # number of columns in column_mask_ == number of expected
        assert sum(ref_column_mask) + isinstance(keep, dict) == exp_num_kept
        assert sum(TestCls.column_mask_) + isinstance(keep, dict) == exp_num_kept
        assert np.array_equal(ref_column_mask, TestCls.column_mask_)
        # assert all columns that werent constants are in the output
        for col_idx in range(_shape[1]):
            if col_idx not in _sorted_constant_idxs:
                assert TestCls.column_mask_[col_idx] is np.True_
        # END column_mask_ ------------


        # constant_columns_ , kept_columns_, removed_columns_ ----------
        # 'constants' that is fed into _X_factory could be None
        ref_constant_columns = constants or {}

        if not equal_nan:
            if has_nan:
                ref_constant_columns = {}
            else:
                ref_constant_columns = {
                    k:v for k,v in ref_constant_columns.items() if str(v) != 'nan'
                }

        for _, __ in (
            (TestCls.constant_columns_, ref_constant_columns),
            (TestCls.kept_columns_, ref_kept_columns),
            (TestCls.removed_columns_, ref_removed_columns)
        ):

            assert np.array_equal(
                sorted(list(_.keys())),
                sorted(list(__.keys()))
            )

            try:
                np.array(list(_.values())).astype(np.float64)
                np.array(list(__.values())).astype(np.float64)
                is_num = True
            except:
                is_num = False

            if is_num:
                assert np.allclose(
                    np.array(list(_.values()), dtype=np.float64),
                    np.array(list(__.values()), dtype=np.float64),
                    equal_nan=True
                )
            else:
                # if values are not num, could be num and str mixed together
                # array_equal is not working in this case. need to iterate
                # over all constant values and check separately.
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
        # END constant_columns_ , kept_columns_, removed_columns_ ------

        # END validate TestCls attrs against ref objects v^v^v^v^v^v^v^v^v^v^v^


        # for retained columns, assert they are equal to themselves in
        # the original data. if the column is not retained, then assert
        # the column in the original data is a constant
        _new_idx = -1
        _kept_idxs = np.arange(len(TestCls.column_mask_))[TestCls.column_mask_]

        if len(_sorted_constant_idxs):
            if keep == 'first':
                assert _sorted_constant_idxs[0] in _kept_idxs
                assert sum([i in _kept_idxs for i in _sorted_constant_idxs]) == 1
            elif keep == 'last':
                assert _sorted_constant_idxs[-1] in _kept_idxs
                assert sum([i in _kept_idxs for i in _sorted_constant_idxs]) == 1
            elif keep == 'random':
                assert sum([i in _kept_idxs for i in _sorted_constant_idxs]) == 1
            elif isinstance(keep, dict) or keep == 'none':
                assert sum([i in _kept_idxs for i in _sorted_constant_idxs]) == 0
        else:
            # if no constants, all columns are kept
            assert np.array_equal(_kept_idxs, range(X.shape[1]))


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









