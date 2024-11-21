# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.InterceptManager.InterceptManager import \
    InterceptManager as IM

from pybear.preprocessing.InterceptManager._partial_fit. \
    _parallel_constant_finder import _parallel_constant_finder


from pybear.preprocessing.ColumnDeduplicateTransformer._partial_fit._parallel_column_comparer import _parallel_column_comparer

from copy import deepcopy
import itertools
import numpy as np
import pandas as pd


import pytest



# pytest.skip(reason=f"pizza is raw!", allow_module_level=True)


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
            'n_jobs': -1     # leave at -1
        }



    @pytest.mark.parametrize('X_format', ('np', )) # pizza 'pd', 'csr', 'csr', 'coo'))
    @pytest.mark.parametrize('X_dtype', ('flt',)) # pizza  'int', 'str', 'obj', 'hybrid'))
    @pytest.mark.parametrize('has_nan', (True, False))
    @pytest.mark.parametrize('constants', (None, )) # pizza {0:1,2:1,9:1}, {0:1,1:1,6:1,8:1}))
    @pytest.mark.parametrize('keep', ('first', 'last', 'random', 'none', 1, 'string', lambda x: 0)) #, {'Intercept': 1})) # pizza
    @pytest.mark.parametrize('equal_nan', (True, False))
    def test_accuracy(self, _X_factory, _kwargs, X_format, X_dtype, has_nan,
        constants, keep, _columns, equal_nan, _shape
    ):

        # validate the test parameters
        assert keep in ['first', 'last', 'random', 'none'] or isinstance(keep, (int, dict, str)) or callable(keep)
        assert isinstance(equal_nan, bool)
        # END validate the test parameters

        if X_dtype in ['str', 'obj', 'hybrid'] and X_format not in ['np', 'pd']:
            pytest.skip(reason=f"scipy sparse cant take str")

        if X_format == 'np' and X_dtype == 'int' and has_nan:
            pytest.skip(reason=f"numpy int dtype cant take 'nan'")


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

        # set _kwargs
        _kwargs['keep'] = keep
        _kwargs['rtol'] = 1e-5
        _kwargs['atol'] = 1e-8
        _kwargs['equal_nan'] = equal_nan
        _kwargs['n_jobs'] = -1


        TestCls = IM(**_kwargs)

        # retain original format
        _og_format = type(X)
        if isinstance(X, pd.core.frame.DataFrame):
            _og_dtype = X.dtypes
        else:
            _og_dtype = X.dtype


        # if data has no constants and user put in keep str/int/callable, will raise
        raise_for_keep_non_constant = False
        has_constants = (constants is not None and len(constants) != 0 and not (has_nan and not equal_nan))
        if not has_constants and (isinstance(keep, (int, str)) or callable(keep)):
            raise_for_keep_non_constant += 1

        if raise_for_keep_non_constant:
            with pytest.raises(ValueError):
                TRFM_X = TestCls.fit_transform(X)
            pytest.skip(f"cannot do more tests without fit")
        else:
            TRFM_X = TestCls.fit_transform(X)
        del raise_for_keep_non_constant




        if isinstance(TRFM_X, np.ndarray):
            assert TRFM_X.flags['C_CONTIGUOUS'] is True

        exp_constants = deepcopy(constants or {})
        if has_nan and not equal_nan:
            exp_constants = {}


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
            assert np.array_equal(TRFM_X.dtypes, _og_dtype[TestCls.column_mask_])
        else:
            assert TRFM_X.dtype == _og_dtype
        # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

        # attr 'n_features_in_' is correct
        assert TestCls.n_features_in_ == X.shape[1]

        # attr 'feature_names_in_' is correct
        if X_dtype == 'pd':
            assert np.array_equal(TestCls.feature_names_in_, _columns)
            assert _columns.dtype == object

        # number of columns in output is adjusted correctly for num constants
        # pizza
        exp_columns = _shape[1]
        if not has_nan or (has_nan and equal_nan):
            if len(exp_constants):
                exp_columns -= len(exp_constants)
                exp_columns +=  (keep != 'none')
            else:
                exp_columns += isinstance(keep, dict)
        elif has_nan and not equal_nan:
            exp_columns += isinstance(keep, dict)
        else:
            raise Exception(f"algorithm failure")
        assert sum(TestCls.column_mask_) == exp_columns
        del exp_columns

        # number of columns in output == number of columns in column_mask_
        assert TRFM_X.shape[1] == sum(TestCls.column_mask_)

        # attr 'constant_columns_' is correct
        if len(exp_constants) == 0:
            assert len(TestCls.constant_columns_) == 0
        else:
            for idx, set in enumerate(exp_constants):
                assert np.array_equal(set, exp_constants[idx])

        # get expected number of kept columns
        _num_kept = X.shape[1] - sum([len(_)-1 for _ in exp_constants])

        # keep ('first','last','random','none') is correct
        # also verify 'column_mask_' 'removed_columns_' 'get_feature_names_out_'
        ref_column_mask = [True for _ in range(X.shape[1])]
        ref_removed_columns = {}
        ref_feature_names_out = list(deepcopy(_columns))


        if keep == 'first':
            for idx in sorted(list(exp_constants.keys()))[1:]:
                ref_column_mask[idx] = False
                ref_removed_columns[idx] = sorted(list(exp_constants.keys()))[0]
                ref_feature_names_out[idx] = None
        elif keep == 'last':
            for idx in sorted(list(exp_constants.keys()))[:-1]:
                ref_column_mask[idx] = False
                ref_removed_columns[idx] = sorted(list(exp_constants.keys()))[-1]
                ref_feature_names_out[idx] = None
        elif keep == 'random':
            # cop out a little here, since we cant know what index
            # was kept, use TestCls.removed_columns_ for help
            # (even tho removed_columns_ is something we are trying
            # to verify)
            ref_removed_columns = deepcopy(TestCls.removed_columns_)
            _kept_idxs = sorted(list(
                np.unique(list(TestCls.removed_columns_.values()))
            ))
            assert len(_kept_idxs) == len(exp_constants)
            _dropped_idxs = sorted(list(
                np.unique(list(TestCls.removed_columns_.keys()))
            ))
            for idx in _dropped_idxs:
                ref_column_mask[idx] = False
                ref_feature_names_out[idx] = None
        elif keep == 'none':
            for idx in sorted(list(exp_constants.keys()))[1:]:
                ref_column_mask[idx] = True
                ref_removed_columns[idx] = {}
                ref_feature_names_out[idx] = None
        elif isinstance(keep, dict):
            for idx in sorted(list(exp_constants.keys()))[:-1]:
                ref_column_mask[idx] = False
                ref_removed_columns[idx] = sorted(list(exp_constants.keys()))[-1]
                ref_feature_names_out[idx] = None
        elif isinstance(keep, int):
            for idx in sorted(list(exp_constants.keys()))[:-1]:
                if idx == keep:
                    ref_column_mask[idx] = True
                    ref_removed_columns[idx] = sorted(list(exp_constants.keys()))[-1]
                    ref_feature_names_out[idx] = None
                else:
                    ref_column_mask[idx] = False
                    ref_removed_columns[idx] = sorted(list(exp_constants.keys()))[-1]
                    ref_feature_names_out[idx] = None
        elif isinstance(keep, str):
            for idx in sorted(list(exp_constants.keys()))[:-1]:
                if keep == TestCls.feature_names_in_[idx]:
                    ref_column_mask[idx] = True
                    ref_removed_columns[idx] = sorted(list(exp_constants.keys()))[-1]
                    ref_feature_names_out[idx] = None
                else:
                    ref_column_mask[idx] = False
                    ref_removed_columns[idx] = sorted(list(exp_constants.keys()))[-1]
                    ref_feature_names_out[idx] = None
        elif callable(keep):
            _callable_keep = keep(X)
            for idx in sorted(list(exp_constants.keys()))[:-1]:
                if idx == _callable_keep:
                    ref_column_mask[idx] = True
                    ref_removed_columns[idx] = sorted(list(exp_constants.keys()))[-1]
                    ref_feature_names_out[idx] = None
                else:
                    ref_column_mask[idx] = False
                    ref_removed_columns[idx] = sorted(list(exp_constants.keys()))[-1]
                    ref_feature_names_out[idx] = None
        else:
            raise Exception


        # for convenient index management, positions to be dropped from
        # ref_feature_names_out were set to None, take those out now
        ref_feature_names_out = [_ for _ in ref_feature_names_out if _ is not None]

        # validate TestCls attrs against ref objects
        assert sum(ref_column_mask) == _num_kept
        assert np.array_equal(ref_column_mask, TestCls.column_mask_)
        if X_dtype == 'pd':
            assert len(TestCls.get_feature_names_out()) == _num_kept
            assert np.array_equal(
                ref_feature_names_out,
                TestCls.get_feature_names_out()
            )
        _, __ = TestCls.removed_columns_, ref_removed_columns
        assert np.array_equal(
            sorted(list(_.keys())),
            sorted(list(__.keys()))
        )
        assert np.array_equal(
            sorted(list(_.values())),
            sorted(list(__.values()))
        )
        del _, __
        # END validate TestCls attrs against ref objects

        # assure all columns that werent constants are in the output
        __all_constants = list(itertools.chain(*deepcopy(exp_constants)))
        for col_idx in range(_shape[1]):
            if col_idx not in __all_constants:
                # pizza when this thing runs come back see if can get this to py True
                assert TestCls.column_mask_[col_idx] is np.True_

        # for retained columns, assert they are equal to themselves in
        # the original data
        _new_idx = -1
        _kept_idxs = np.arange(len(TestCls.column_mask_))[TestCls.column_mask_]
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

            if _idx in _kept_idxs:
                assert _parallel_column_comparer(
                    _out_col,
                    _og_col,
                    _equal_nan=True,
                    _rtol=1e-5,
                    _atol=1e-8,
                )
            else:
                _parallel_constant_finder(
                    _column=_out_col,
                    _equal_nan=equal_nan,
                    _rtol=1e-5,
                    _atol=1e-8
                )

        # END ASSERTIONS ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *









