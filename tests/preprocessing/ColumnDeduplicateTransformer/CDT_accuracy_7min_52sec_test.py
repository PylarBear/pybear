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







pytest.skip(reason=f"pizza says so", allow_module_level=True)




# TEST ACCURACY ********************************************************


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

        # retain original columns
        _og_cols = X.shape[1]

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
        # returned format is same as given format
        assert isinstance(TRFM_X, _og_format)

        # returned dtypes are same as given dtypes
        if isinstance(TRFM_X, pd.core.frame.DataFrame):
            assert np.array_equal(TRFM_X.dtypes, _og_dtype[TestCls.column_mask_])
        else:
            assert TRFM_X.dtype == _og_dtype

        # number of columns in output is adjusted correctly for num duplicates
        assert sum(TestCls.column_mask_) == \
               _shape[1] - sum([len(_)-1 for _ in exp_dupls])

        # number of columns in output == number of columns in column_mask_
        assert TRFM_X.shape[1] == sum(TestCls.column_mask_)

        _kept_idxs = np.arange(_shape[1])[TestCls.column_mask_]

        # keep ('first','last','random') is correct when not muddied by do_not_drop
        if not _conflict_condition:
            # in this case, column idxs cannot be overridden by do_not_drop
            if keep == 'first':
                for _set in exp_dupls:
                    assert _set[0] in _kept_idxs
            elif keep == 'last':
                for _set in exp_dupls:
                    assert _set[-1] in _kept_idxs
            elif keep == 'random':
                for _set in exp_dupls:
                    assert sum([__ in _kept_idxs for __ in _set]) == 1
        elif _conflict_condition and conflict == 'raise':
            # this should have raised above
            raise Exception
        elif _conflict_condition and conflict == 'ignore':
            # this could get really complicated. suffice it to determine
            # only one column from each group of duplicates is kept
            for _set in exp_dupls:
                assert sum([__ in _kept_idxs for __ in _set]) == 1
        else:
            raise Exception

        # assure all columns that werent duplicates are in the output
        __all_dupls = list(itertools.chain(*deepcopy(exp_dupls)))
        for col_idx in range(_shape[1]):
            if col_idx not in __all_dupls:
                assert col_idx in _kept_idxs

        # for retained columns, assert they are equal to themselves in
        # the original data
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

# END TEST ACCURACY ****************************************************







