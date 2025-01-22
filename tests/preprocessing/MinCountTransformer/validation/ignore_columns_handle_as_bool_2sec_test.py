# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.MinCountTransformer._validation. \
    _ignore_columns_handle_as_bool import _val_ignore_columns_handle_as_bool

import numpy as np
import pandas as pd

import pytest



class TestValIgnoreColumns:

    # def _val_ignore_columns_handle_as_bool(
    #     _value: Union[IgnoreColumnsType, HandleAsBoolType],
    #     _name: Literal['ignore_columns', 'handle_as_bool'],
    #     _n_features_in: int,
    #     _feature_names_in: Union[npt.NDArray[str], None]=None
    # ) -> None:


    # helper param validation ** * ** * ** * ** * ** * ** * ** * ** * **

    # _name -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    @pytest.mark.parametrize('junk_name',
        (np.pi, True, min, {'a':1}, lambda x: x)
    )
    def test_reject_junk_name(self, junk_name):
        with pytest.raises(AssertionError):
            _val_ignore_columns_handle_as_bool(
                [0, 1, 2, 3],
                junk_name,
                4,
                None
            )


    @pytest.mark.parametrize('bad_name', ('lemon', 'lime', 'orange'))
    def test_reject_bad_name(self, bad_name):
        with pytest.raises(AssertionError):
            _val_ignore_columns_handle_as_bool(
                [0, 1, 2, 3],
                bad_name,
                4,
                None
            )

    # END _name -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    # n_features_in_ -- -- -- -- -- -- -- -- -- -- -- -- --

    @pytest.mark.parametrize('good_value',
        (['a', 'b', 'c', 'c'], [0, 1, 2, 3], lambda x: x, None)
    )
    def test_n_features_in_cannot_be_None(self, good_value):
        with pytest.raises(AssertionError):
            _val_ignore_columns_handle_as_bool(
                good_value,
                'ignore_columns',
                None,
                None
            )


    @pytest.mark.parametrize('junk_nfi',
        (np.pi, True, min, [0,1], (0,1), {0,1}, {'a':1}, lambda x: x)
    )
    def test_reject_junk_n_features_in(self, junk_nfi):
        with pytest.raises(AssertionError):
            _val_ignore_columns_handle_as_bool(
                None,
                'handle_as_bool',
                junk_nfi,
                None
            )


    def test_reject_bad_n_features_in(self):
        with pytest.raises(AssertionError):
            _val_ignore_columns_handle_as_bool(
                [0, 1, 2, 3],
                'ignore_colums',
                -1,
                None
            )

    # END n_features_in_ -- -- -- -- -- -- -- -- -- -- -- -- --

    # feature_names_in_ -- -- -- -- -- -- -- -- -- -- -- -- -- --
    @pytest.mark.parametrize('junk_fni',
        (np.pi, True, min, {'a':1}, lambda x: x)
    )
    def test_reject_junk_feature_names_in(self, junk_fni):
        with pytest.raises(AssertionError):
            _val_ignore_columns_handle_as_bool(
                [0, 1, 2, 3],
                'handle_as_bool',
                4,
                junk_fni
            )


    @pytest.mark.parametrize('bad_fni', ([0, 1], (0, 1), {0, 1}))
    def test_reject_bad_feature_names_in(self, bad_fni):
        with pytest.raises(AssertionError):
            _val_ignore_columns_handle_as_bool(
                [0, 1, 2, 3],
                'ignore_columns',
                4,
                bad_fni
            )


    @pytest.mark.parametrize('_fni', (list('abcd'), set('abcd'), tuple('abcd')))
    def test_feature_names_in_must_be_np(self, _fni):

        with pytest.raises(AssertionError):
            _val_ignore_columns_handle_as_bool(
                None,
                'handle_as_bool',
                _n_features_in=4,
                _feature_names_in=_fni
            )

    # END feature_names_in_ -- -- -- -- -- -- -- -- -- -- -- -- --

    # joint nfi fni  -- -- -- -- -- -- -- -- -- -- -- -- --
    @pytest.mark.parametrize('bad_nfi', (3,5))
    def test_value_error_nfi_not_equal_len_fni(self, bad_nfi):

        with pytest.raises(AssertionError):
            _val_ignore_columns_handle_as_bool(
                [0, 1, 2, 3],
                'ignore_columns',
                bad_nfi,
                _feature_names_in = np.array(['a', 'b', 'c', 'd'], dtype='<U1')
            )

    # END joint nfi fni  -- -- -- -- -- -- -- -- -- -- -- -- --

    # END helper param validation ** * ** * ** * ** * ** * ** * ** * ** * **



    @pytest.mark.parametrize('junk_value',
        (-2.7, -1, 0, 1, 2.7, np.pi, True, 'junk', {'a': 1})
    )
    def test_rejects_junk_value(self, junk_value):

        with pytest.raises(TypeError):
            _val_ignore_columns_handle_as_bool(
                junk_value,
                'handle_as_bool',
                _n_features_in=4,
                _feature_names_in=np.array(list('abcd'))
            )


    def test_accepts_empty_listlike_value(self):

        _val_ignore_columns_handle_as_bool(
            [],
            'ignore_columns',
            _n_features_in=4,
            _feature_names_in=np.array(list('abcd'))
        )


    @pytest.mark.parametrize('fxn_type', ('def', 'lambda'))
    def test_ignore_columns_accepts_callable(self, fxn_type):

        def _dum_fxn(X):
            return [0,1]

        if fxn_type == 'def':
            _fxn = _dum_fxn
        elif fxn_type == 'lambda':
            _fxn = lambda X: [0, 1]
        else:
            raise Exception

        out = _val_ignore_columns_handle_as_bool(
            _fxn,
            'handle_as_bool',
            _n_features_in=4,
            _feature_names_in= np.array(['a', 'b', 'c', 'd'], dtype='<U1')
        )

        assert out is None


    @pytest.mark.parametrize('_fni',
        (None, np.array(['a', 'b', 'c', 'd'], dtype='<U1'))
    )
    def test_value_accepts_None(self, _fni):

        out = _val_ignore_columns_handle_as_bool(
            None,
            'ignore_columns',
            _n_features_in=4,
            _feature_names_in=_fni
        )

        assert out is None


    @pytest.mark.parametrize('list_like',
        (list('abcd'), set('abcd'), tuple('abcd'),
        np.array(list('abcd'), dtype='<U1'))
    )
    def test_value_accepts_list_like(self, list_like):
        out = _val_ignore_columns_handle_as_bool(
            list_like,
            'handle_as_bool',
            _n_features_in=4,
            _feature_names_in=np.array(['a', 'b', 'c', 'd'], dtype='<U1')
        )

        assert out is None


    @pytest.mark.parametrize('duplicate_values', (list('aacd'), (0, 1, 2, 2, 3)))
    def test_rejects_duplicate_values(self, duplicate_values):
        with pytest.raises(ValueError):
            _val_ignore_columns_handle_as_bool(
                duplicate_values,
                'handle_as_bool',
                _n_features_in=4,
                _feature_names_in=np.array(['a', 'b', 'c', 'd'], dtype='<U1')
            )


    # test values in list-like ** * ** * ** * ** * ** * ** * ** * ** * **
    @pytest.mark.parametrize('junk_value',
        (True, False, np.pi, None, min, [0,1], (0,1), {0,1}, {'a':1}, lambda x: x)
    )
    @pytest.mark.parametrize('_fni', (None, np.array(['a'])))
    def test_rejects_non_int_not_str_in_list_type(self, junk_value, _fni):
        with pytest.raises(TypeError):
            _val_ignore_columns_handle_as_bool(
                [junk_value],
                'ignore_columns',
                _n_features_in=1,
                _feature_names_in=_fni
            )


    @pytest.mark.parametrize('_fni', (None, np.array(list('abcde'))))
    def test_rejects_nanlike_in_list_type(self, _fni):

        with pytest.raises(TypeError):
            _val_ignore_columns_handle_as_bool(
                [0, 1, np.nan],
                'ignore_columns',
                _n_features_in=5,
                _feature_names_in=_fni
            )


        with pytest.raises(TypeError):
            _val_ignore_columns_handle_as_bool(
                [pd.NA, 2],
                'ignore_columns',
                _n_features_in=5,
                _feature_names_in=_fni
            )


    @pytest.mark.parametrize('_fni', (None, np.array(list('abcd'))))
    def test_reject_list_types_of_different_types(self, _fni):

        with pytest.raises(TypeError):
            _val_ignore_columns_handle_as_bool(
                ['a', 'b', 3, 4],
                'handle_as_bool',
                _n_features_in=4,
                _feature_names_in=_fni
            )


    def test_value_error_idx_out_of_bounds(self):

        with pytest.raises(ValueError):
            _val_ignore_columns_handle_as_bool(
                [0, 1, 2, 3],
                'ignore_columns',
                3,
                None
            )

        with pytest.raises(ValueError):
            _val_ignore_columns_handle_as_bool(
                [-4, -3, -2],
                'handle_as_bool',
                3,
                None
            )


    def test_value_error_column_names_passed_when_no_fni(self):

        with pytest.raises(ValueError):
            _val_ignore_columns_handle_as_bool(
                ['a', 'b', 'c', 'd'],
                'ignore_columns',
                4,
                None
            )


    def test_value_error_col_name_not_in_feature_names(self):

        feature_names_in = np.array(['a', 'b', 'c', 'd'], dtype='<U1')

        with pytest.raises(ValueError):
            _val_ignore_columns_handle_as_bool(
                ['e'],
                'handle_as_bool',
                4,
                feature_names_in
            )

    # END test values in list-like ** * ** * ** * ** * ** * ** * ** * ** * **






















