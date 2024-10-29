# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.ColumnDeduplicateTransformer. \
    ColumnDeduplicateTransformer import ColumnDeduplicateTransformer as CDT

from pybear.preprocessing.ColumnDeduplicateTransformer._partial_fit. \
    _parallel_column_comparer import _parallel_column_comparer


from copy import deepcopy
import itertools
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
            'do_not_drop': None,
            'conflict': 'raise',
            'rtol': 1e-5,
            'atol': 1e-8,
            'equal_nan': False,
            'n_jobs': -1     # leave at -1
        }



    @pytest.mark.parametrize('X_format', ('np', 'pd', 'csr')) #, 'csr', 'coo'))
    @pytest.mark.parametrize('X_dtype', ('flt', 'int', 'str', 'obj', 'hybrid'))
    @pytest.mark.parametrize('has_nan', (True, False))
    @pytest.mark.parametrize('dupls', (None, [[0,2,9]], [[0,6],[1,8]]))
    @pytest.mark.parametrize('keep', ('first', 'last', 'random'))
    @pytest.mark.parametrize('do_not_drop', (None, [0,5], 'pd'))
    @pytest.mark.parametrize('conflict', ('raise', 'ignore'))
    @pytest.mark.parametrize('equal_nan', (True, False))
    def test_accuracy(self, _X_factory, _kwargs, X_format, X_dtype, has_nan,
        dupls, keep, do_not_drop, conflict, _columns, equal_nan, _shape
    ):

        # validate the test parameters
        assert keep in ['first', 'last', 'random']
        assert isinstance(do_not_drop, (list, type(None), str))
        assert conflict in ['raise', 'ignore']
        assert isinstance(equal_nan, bool)
        # END validate the test parameters

        if X_dtype in ['str', 'obj', 'hybrid'] and X_format not in ['np', 'pd']:
            pytest.skip(reason=f"scipy sparse cant take str")

        if X_format == 'np' and X_dtype == 'int' and has_nan:
            pytest.skip(reason=f"numpy int dtype cant take 'nan'")

        if do_not_drop == 'pd':
            if X_format != 'pd':
                pytest.skip(
                    reason=f"impossible condition, str dnd and format is not pd"
                )


        X = _X_factory(
            _dupl=dupls,
            _format=X_format,
            _dtype=X_dtype,
            _has_nan=has_nan,
            _shape=_shape,
            _columns=_columns
        )

        # set do_not_drop as list of strings (vs None or list of ints)
        if do_not_drop == 'pd':
            do_not_drop = list(map(str, [_columns[0], _columns[3], _columns[7]]))

        # do the conflict conditions exist?
        _conflict_condition = (dupls is not None) and (do_not_drop is not None) \
            and (keep == 'last') and not (has_nan and not equal_nan)
        # only because all non-None dupls and non-None do_not_drop
        # have zeros in them

        # set _kwargs
        _kwargs['keep'] = keep
        _kwargs['do_not_drop'] = do_not_drop
        _kwargs['conflict'] = conflict
        _kwargs['rtol'] = 1e-5
        _kwargs['atol'] = 1e-8
        _kwargs['equal_nan'] = equal_nan
        _kwargs['n_jobs'] = -1


        TestCls = CDT(**_kwargs)

        # retain original format
        _og_format = type(X)
        if isinstance(X, pd.core.frame.DataFrame):
            _og_dtype = X.dtypes
        else:
            _og_dtype = X.dtype

        if _conflict_condition and conflict=='raise':
            with pytest.raises(ValueError):
                TestCls.fit_transform(X)
            pytest.skip(reason=f"dont do remaining tests")
        else:
            TRFM_X = TestCls.fit_transform(X)

        exp_dupls = deepcopy(dupls or [])
        if has_nan and not equal_nan:
            exp_dupls = []


        # ASSERTIONS ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

        # attributes:
        #     'n_features_in_'
        #     'feature_names_in_'
        #     'duplicates_'
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

        # number of columns in output is adjusted correctly for num duplicates
        assert sum(TestCls.column_mask_) == \
               _shape[1] - sum([len(_)-1 for _ in exp_dupls])

        # number of columns in output == number of columns in column_mask_
        assert TRFM_X.shape[1] == sum(TestCls.column_mask_)

        # attr 'duplicates_' is correct
        if len(exp_dupls) == 0:
            assert len(TestCls.duplicates_) == 0
        else:
            for idx, set in enumerate(exp_dupls):
                assert np.array_equal(set, exp_dupls[idx])

        # get expected number of kept columns
        _num_kept = X.shape[1] - sum([len(_)-1 for _ in exp_dupls])

        # keep ('first','last','random') is correct when not muddied by do_not_drop
        # also verify 'column_mask_' 'removed_columns_' 'get_feature_names_out_'
        ref_column_mask = [True for _ in range(X.shape[1])]
        ref_removed_columns = {}
        ref_feature_names_out = list(deepcopy(_columns))
        if not _conflict_condition:
            # in this case, column idxs cannot be overridden by do_not_drop
            for _set in exp_dupls:
                if keep == 'first':
                    for idx in _set[1:]:
                        ref_column_mask[idx] = False
                        ref_removed_columns[idx] = _set[0]
                        ref_feature_names_out[idx] = None
                elif keep == 'last':
                    for idx in _set[:-1]:
                        ref_column_mask[idx] = False
                        ref_removed_columns[idx] = _set[-1]
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
                    assert len(_kept_idxs) == len(exp_dupls)
                    _dropped_idxs = sorted(list(
                        np.unique(list(TestCls.removed_columns_.keys()))
                    ))
                    for idx in _dropped_idxs:
                        ref_column_mask[idx] = False
                        ref_feature_names_out[idx] = None

        elif _conflict_condition and conflict == 'raise':
            # this should have raised above
            raise Exception
        elif _conflict_condition and conflict == 'ignore':
            # this could get really complicated.
            # cop out here again, since the logic behind what index
            # was kept is really complicated, use TestCls.removed_columns_
            # for help (even tho removed_columns_ is something we are trying
            # to verify)
            ref_removed_columns = deepcopy(TestCls.removed_columns_)
            _kept_idxs = sorted(list(
                np.unique(list(TestCls.removed_columns_.values()))
            ))
            assert len(_kept_idxs) == len(exp_dupls)
            _dropped_idxs = sorted(list(
                np.unique(list(TestCls.removed_columns_.keys()))
            ))
            for idx in _dropped_idxs:
                ref_column_mask[idx] = False
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

        # assure all columns that werent duplicates are in the output
        __all_dupls = list(itertools.chain(*deepcopy(exp_dupls)))
        for col_idx in range(_shape[1]):
            if col_idx not in __all_dupls:
                assert TestCls.column_mask_[col_idx] is True

        # for retained columns, assert they are equal to themselves in
        # the original data
        _kept_idxs = np.arange(len(TestCls.column_mask_))[TestCls.column_mask_]
        for _new_idx, _kept_idx in enumerate(_kept_idxs, 0):

            if isinstance(X, np.ndarray):
                _out_col = TRFM_X[:, _new_idx]
                _og_col = X[:, _kept_idx]
            elif isinstance(X, pd.core.frame.DataFrame):
                _out_col = TRFM_X.iloc[:, _new_idx].to_numpy()
                _og_col = X.iloc[:, _kept_idx].to_numpy()
            else:
                _out_col = TRFM_X.tocsc()[:, [_new_idx]].toarray()
                _og_col = X.tocsc()[:, [_kept_idx]].toarray()


            assert _parallel_column_comparer(
                _out_col,
                _og_col,
                _rtol=1e-5,
                _atol=1e-8,
                _equal_nan=True
            )

        # END ASSERTIONS ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *









