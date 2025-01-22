# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.MinCountTransformer._validation. \
    _core_ign_cols_handle_as_bool import _core_ign_cols_handle_as_bool

import numpy as np

import pytest



class TestCoreIgnColsHandleAsBool:

    # def _core_ign_cols_handle_as_bool(
    #     _kwarg_value: Union[IgnoreColumnsType, HandleAsBoolType],
    #     _name: str,
    #     _mct_has_been_fit: bool=False,
    #     _n_features_in: Union[None, int]=None,
    #     _feature_names_in: Union[npt.NDArray[str], None]=None
    # ) -> Union[callable, npt.NDArray[int], npt.NDArray[str]]:

    # vvv functionality independent of fit or not vvv ** * ** * ** * ** *

    @pytest.mark.parametrize('junk_kwarg_value',
        (0, 1, np.pi, True, 'junk', {'a':1})
    )
    @pytest.mark.parametrize('_mct_has_been_fit, _n_features_in, _fni',
        (
            (False, None, None),
            (True, 3, None),
            (True, 4, ['a', 'b', 'c', 'd']),
        )
    )
    def test_rejects_non_None_callable_list_like(self, junk_kwarg_value,
        _mct_has_been_fit, _n_features_in, _fni):

        with pytest.raises(TypeError):
            _core_ign_cols_handle_as_bool(
                junk_kwarg_value,
                'ignore_columns',
                _mct_has_been_fit,
                _n_features_in,
                _fni
            )


    @pytest.mark.parametrize('bad_name', ('bacon', 'lettuce', 'tomato'))
    def test_rejects_bad_name(self, bad_name):

        with pytest.raises(ValueError):
            _core_ign_cols_handle_as_bool(
                [0,1],
                _name=bad_name,
                _mct_has_been_fit=True,
                _n_features_in=4,
                _feature_names_in=None
            )


    @pytest.mark.parametrize('_mct_has_been_fit, _n_features_in, _fni',
        (
            (False, None, None),
            (True, 3, None),
            (True, 4, np.array(['a', 'b', 'c', 'd'], dtype='<U1')),
        )
    )
    def test_callable_returns_callable(
        self, _mct_has_been_fit, _n_features_in, _fni
    ):

        out = _core_ign_cols_handle_as_bool(
            lambda x: x,
            'handle_as_bool',
            _mct_has_been_fit,
            _n_features_in,
            _fni
        )

        assert callable(out)
        assert out(3) == 3


    @pytest.mark.parametrize('_mct_has_been_fit, _n_features_in, _fni',
        (
            (False, None, None),
            (True, 3, None),
            (True, 4, np.array(['a', 'b', 'c', 'd'], dtype='<U1')),
        )
    )
    def test_None_returns_empty_ndnarray(
        self, _mct_has_been_fit, _n_features_in, _fni
    ):
        out = _core_ign_cols_handle_as_bool(
            None,
            'ignore_columns',
            _mct_has_been_fit,
            _n_features_in,
            _fni
        )

        assert isinstance(out, np.ndarray)
        assert len(out) == 0


    @pytest.mark.parametrize('array_like',
        (
            ['a', 'b', 'c', 'd'],
            {'a', 'b', 'c', 'd'},
            ('a', 'b', 'c', 'd'),
            np.array(['a', 'b', 'c', 'd'], dtype='<U1')
        )
    )
    @pytest.mark.parametrize('_mct_has_been_fit, _n_features_in, _fni',
        (
            (False, None, None),
            (True, 4, np.array(['a', 'b', 'c', 'd'], dtype='<U1')),
        )
    )
    def test_accepts_list_like(
        self, array_like, _mct_has_been_fit, _n_features_in, _fni
    ):
        out = _core_ign_cols_handle_as_bool(
            array_like,
            'handle_as_bool',
            _mct_has_been_fit,
            _n_features_in,
            _fni
        )

        # if hasnt been fit, return list-like that was passed as ndarray
        # if is fit, return ndarray of indices

        assert isinstance(out, np.ndarray)

        if _mct_has_been_fit:
            assert np.array_equiv(out, [0, 1, 2, 3])
        elif not _mct_has_been_fit:
            assert np.array_equiv(out, ['a', 'b', 'c', 'd'])


    @pytest.mark.parametrize('junk_value',
        (True, False, np.pi, None, min, [0,1], (0,1), {0,1}, {'a':1}, lambda x: x)
    )
    @pytest.mark.parametrize('_mct_has_been_fit, _n_features_in, _fni',
        (
            (False, None, None),
            (True, 1, None),
            (True, 1, ['a']),
        )
    )
    def test_rejects_non_int_not_str_in_list_type(
        self, junk_value, _mct_has_been_fit, _n_features_in, _fni
    ):

        with pytest.raises(TypeError):
            _core_ign_cols_handle_as_bool(
                [junk_value],
                'ignore_columns',
                _mct_has_been_fit,
                _n_features_in,
                _fni
            )


    @pytest.mark.parametrize('_mct_has_been_fit, _n_features_in, _fni',
        (
            (False, None, None),
            (True, 4, None),
            (True, 4, ['a', 'b', 'c', 'd']),
        )
    )
    def test_reject_list_types_of_different_types(
        self, _mct_has_been_fit, _n_features_in, _fni
    ):
        with pytest.raises(TypeError):
            _core_ign_cols_handle_as_bool(
                ['a', 'b', 3, 4],
                'handle_as_bool',
                _mct_has_been_fit,
                _n_features_in,
                _fni
            )

    # ^^^ END functionality independent of fit or not ^^^ ** * ** * ** *


    # vvv functionality when fit vvv ** * ** * ** * ** * ** * ** * ** *

    @pytest.mark.parametrize('good_kwarg_value',
        (['a', 'b', 'c', 'c'], [0, 1, 2, 3], lambda x: x, None)
    )
    def test_if_fit_must_pass_n_features_in(self, good_kwarg_value):
        with pytest.raises(ValueError):
            _core_ign_cols_handle_as_bool(
                good_kwarg_value, 'ignore_columns', True, None, None
            )


    @pytest.mark.parametrize('good_kwarg_value',
        (['a', 'b', 'c', 'c'], [0, 1, 2, 3], lambda x: x, None)
    )
    def test_if_pass_n_features_in_must_be_fit(self, good_kwarg_value):
        with pytest.raises(ValueError):
            _core_ign_cols_handle_as_bool(
                good_kwarg_value, 'handle_as_bool', False, 4, None
            )


    @pytest.mark.parametrize('junk_nfi',
        (np.pi, True, min, [0,1], (0,1), {0,1}, {'a':1}, lambda x: x)
    )
    def test_reject_junk_n_features_in(self, junk_nfi):
        with pytest.raises(AssertionError):
            _core_ign_cols_handle_as_bool(
                [0, 1, 2, 3], 'ignore_columns', True, junk_nfi, None
            )


    def test_reject_bad_n_features_in(self):
        with pytest.raises(AssertionError):
            _core_ign_cols_handle_as_bool(
                [0, 1, 2, 3], 'handle_as_bool', True, -1, None
            )


    def test_value_error_idx_gt_n_features_in(self):
        with pytest.raises(ValueError):
            _core_ign_cols_handle_as_bool(
                [0, 1, 2, 3], 'ignore_columns', True, 3, None
            )


    @pytest.mark.parametrize('bad_nfi', (3,5))
    def test_value_error_nfi_not_equal_len_fni(self, bad_nfi):

        _feature_names_in = np.array(['a', 'b', 'c', 'd'], dtype='<U1')

        with pytest.raises(ValueError):
            _core_ign_cols_handle_as_bool(
                [0, 1, 2, 3], 'handle_as_bool', True, bad_nfi, _feature_names_in
            )


    def test_value_error_fni_passed_when_not_fit(self):

        feature_names_in = np.array(['a', 'b', 'c', 'd'], dtype='<U1')

        with pytest.raises(ValueError):
            _core_ign_cols_handle_as_bool(
                ['a'], 'ignore_columns', False, 4, feature_names_in
            )


    def test_value_error_col_name_not_in_feature_names(self):

        feature_names_in = np.array(['a', 'b', 'c', 'd'], dtype='<U1')

        with pytest.raises(ValueError):
            _core_ign_cols_handle_as_bool(
                ['e'], 'handle_as_bool', True, 4, feature_names_in
            )


    def test_value_error_column_names_passed_when_no_fni(self):

        with pytest.raises(ValueError):
            _core_ign_cols_handle_as_bool(
                ['a', 'b', 'c', 'd'], 'ignore_columns', True, 4, None
            )

    # ^^^ END functionality when fit ^^^ ** * ** * ** * ** * ** * ** * **






























