# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest
import numpy as np

from preprocessing.MinCountTransformer._shared._validation._val_ignore_columns \
    import _val_ignore_columns



class TestCoreIgnColsHandleAsBool:

    # def _core_ign_cols_handle_as_bool(
    #         kwarg_value: IgnColsHandleAsBoolDtype,
    #         _mct_has_been_fit: bool = False,
    #         _n_features_in: Union[None, int] = None,
    #         _feature_names_in: Union[None, np.ndarray[str]] = None
    # ) -> Union[Callable, np.ndarray[int], np.ndarray[str]]:

    # vvv functionality independent of fit or not vvv ** * ** * ** * ** *

    @pytest.mark.parametrize('junk_kwarg_value',
        (0, 1, np.pi, True, 'junk', {'a':1})
    )
    @pytest.mark.parametrize('_has_been_fit, _n_features_in, _fni',
        (
            (False, None, None),
            (True, 3, None),
            (True, 4, ['a', 'b', 'c', 'd']),
        )
    )
    def test_rejects_non_None_callable_list_like(self, junk_kwarg_value,
        _has_been_fit, _n_features_in, _fni):

        with pytest.raises(TypeError):
            _val_ignore_columns(
                junk_kwarg_value,
                _has_been_fit,
                _n_features_in,
                _fni
            )


    @pytest.mark.parametrize('_has_been_fit, _n_features_in, _fni',
        (
            (False, None, None),
            (True, 3, None),
            (True, 4, np.array(['a', 'b', 'c', 'd'], dtype='<U1')),
        )
    )
    def test_callable_returns_callable(self, _has_been_fit, _n_features_in,
                                       _fni):
        out = _val_ignore_columns(
            lambda x: x,
            _has_been_fit,
            _n_features_in,
            _fni
        )

        assert callable(out)
        assert out(3) == 3


    @pytest.mark.parametrize('_has_been_fit, _n_features_in, _fni',
        (
            (False, None, None),
            (True, 3, None),
            (True, 4, np.array(['a', 'b', 'c', 'd'], dtype='<U1')),
        )
    )
    def test_None_returns_empty_ndnarray(self, _has_been_fit, _n_features_in,
                                         _fni):
        out = _val_ignore_columns(
            None,
            _has_been_fit,
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
    @pytest.mark.parametrize('_has_been_fit, _n_features_in, _fni',
        (
            (False, None, None),
            (True, 4, np.array(['a', 'b', 'c', 'd'], dtype='<U1')),
        )
    )
    def test_accepts_list_like(self, array_like, _has_been_fit, _n_features_in,
                               _fni):
        out = _val_ignore_columns(
            array_like,
            _has_been_fit,
            _n_features_in,
            _fni
        )

        # if hasnt been fit, return list-like that was passed as ndarray
        # if is fit, return ndarray of indices

        assert isinstance(out, np.ndarray)

        if _has_been_fit:
            assert np.array_equiv(out, [0, 1, 2, 3])
        elif not _has_been_fit:
            assert np.array_equiv(out, ['a', 'b', 'c', 'd'])


    @pytest.mark.parametrize('junk_value',
        (True, False, np.pi, None, min, [0,1], (0,1), {0,1}, {'a':1}, lambda x: x)
    )
    @pytest.mark.parametrize('_has_been_fit, _n_features_in, _fni',
        (
            (False, None, None),
            (True, 1, None),
            (True, 1, ['a']),
        )
    )
    def test_rejects_non_int_not_str_in_list_type(self, junk_value,
        _has_been_fit, _n_features_in, _fni):
        with pytest.raises(TypeError):
            _val_ignore_columns(
                [junk_value],
                _has_been_fit,
                _n_features_in,
                _fni
            )


    @pytest.mark.parametrize('_has_been_fit, _n_features_in, _fni',
        (
            (False, None, None),
            (True, 4, None),
            (True, 4, ['a', 'b', 'c', 'd']),
        )
    )
    def test_reject_list_types_of_different_types(self, _has_been_fit,
                                                  _n_features_in, _fni):
        with pytest.raises(TypeError):
            _val_ignore_columns(
                ['a', 'b', 3, 4],
                _has_been_fit,
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
            _val_ignore_columns(good_kwarg_value, True, None, None)


    @pytest.mark.parametrize('good_kwarg_value',
        (['a', 'b', 'c', 'c'], [0, 1, 2, 3], lambda x: x, None)
    )
    def test_if_pass_n_features_in_must_be_fit(self, good_kwarg_value):
        with pytest.raises(ValueError):
            _val_ignore_columns(good_kwarg_value, False, 4, None)


    @pytest.mark.parametrize('junk_nfi',
        (np.pi, True, min, [0,1], (0,1), {0,1}, {'a':1}, lambda x: x)
    )
    def test_reject_junk_n_features_in(self, junk_nfi):
        with pytest.raises(TypeError):
            _val_ignore_columns([0, 1, 2, 3], True, junk_nfi, None)


    def test_reject_bad_n_features_in(self):
        with pytest.raises(ValueError):
            _val_ignore_columns([0, 1, 2, 3], True, -1, None)


    def test_value_error_idx_gt_n_features_in(self):
        with pytest.raises(ValueError):
            _val_ignore_columns([0, 1, 2, 3], True, 3, None)


    @pytest.mark.parametrize('bad_nfi', (3,5))
    def test_value_error_nfi_not_equal_len_fni(self, bad_nfi):

        _feature_names_in = np.array(['a', 'b', 'c', 'd'], dtype='<U1')

        with pytest.raises(ValueError):
            _val_ignore_columns([0, 1, 2, 3], True, bad_nfi, _feature_names_in)


    def test_value_error_fni_passed_when_not_fit(self):

        feature_names_in = np.array(['a', 'b', 'c', 'd'], dtype='<U1')

        with pytest.raises(ValueError):
            _val_ignore_columns(['a'], False, 4, feature_names_in)


    def test_value_error_col_name_not_in_feature_names(self):

        feature_names_in = np.array(['a', 'b', 'c', 'd'], dtype='<U1')

        with pytest.raises(ValueError):
            _val_ignore_columns(['e'], True, 4, feature_names_in)


    def test_value_error_column_names_passed_when_no_fni(self):

        with pytest.raises(ValueError):
            _val_ignore_columns(['a', 'b', 'c', 'd'], True, 4, None)

    # ^^^ END functionality when fit ^^^ ** * ** * ** * ** * ** * ** * **






























