# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.MinCountTransformer._validation.\
    _ign_cols_hab_callable import _val_ign_cols_hab_callable

import numpy as np

import pytest



class TestValIgnColsHabCallable:

    # def _val_ign_cols_hab_callable(
    #     _fxn_output: Union[Iterable[str], Iterable[int]],
    #     _name: Literal['ignore_columns', 'handle_as_bool'],
    #     _n_features_in: int,
    #     _feature_names_in: Union[npt.NDArray[str], None]
    # ) -> None:

    # helper object validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # validate _n_features_in -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    @pytest.mark.parametrize('junk_n_features_in',
        (-2.7, 2.7, True, None, 'garbage', [0, 1], (1,), {'a':1}, lambda x: x)
    )
    def test_rejects_junk_n_features_in(self, junk_n_features_in):
        with pytest.raises(TypeError):
            _val_ign_cols_hab_callable(
                [0, 1],
                'ignore_columns',
                junk_n_features_in,
                None
            )


    @pytest.mark.parametrize('bad_n_features_in', (-1, 0))
    def test_rejects_bad_n_features_in(self, bad_n_features_in):
        with pytest.raises(ValueError):
            _val_ign_cols_hab_callable(
                [0, 1],
                'handle_as_bool',
                bad_n_features_in,
                None
            )


    @pytest.mark.parametrize('good_n_features_in', (2, 10, 1_000))
    def test_accepts_good_n_features_in(self, good_n_features_in):
        _val_ign_cols_hab_callable(
            [0, 1],
            'ignore_columns',
            good_n_features_in,
            None
        )
    # END validate _n_features_in -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    # validate _name -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    @pytest.mark.parametrize('junk_name',
        (-2.7, -1, 0, 1, 2.7, True, None, [0, 1], (1,), {'a':1}, lambda x: x)
    )
    def test_rejects_junk_name(self, junk_name):
        with pytest.raises(TypeError):
            _val_ign_cols_hab_callable(
                [0, 1],
                junk_name,
                10,
                None
            )


    @pytest.mark.parametrize('bad_name', ('wings', 'chips', 'beer'))
    def test_rejects_bad_name(self, bad_name):
        with pytest.raises(ValueError):
            _val_ign_cols_hab_callable(
                [0, 1],
                bad_name,
                10,
                None
            )


    @pytest.mark.parametrize('good_name', ('ignore_columns', 'handle_as_bool'))
    def test_accepts_good_name(self, good_name):
        _val_ign_cols_hab_callable(
            [0, 1],
            good_name,
            10,
            None
        )
    # END validate _name -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    # validate feature_names_in -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    @pytest.mark.parametrize('junk_fni',
        (-2.7, -1, 0, 1, 2.7, True, 'trash', {'a':1}, lambda x: x)
    )
    def test_rejects_junk_fni(self, junk_fni):
        with pytest.raises(TypeError):
            _val_ign_cols_hab_callable(
                [0, 1],
                'ignore_columns',
                10,
                junk_fni
            )


    @pytest.mark.parametrize('bad_fni', ([0,1,2], (True, False), {-3, -2, -1}))
    def test_rejects_bad_fni(self, bad_fni):
        with pytest.raises(ValueError):
            _val_ign_cols_hab_callable(
                [0, 1],
                'handle_as_bool',
                10,
                bad_fni
            )


    @pytest.mark.parametrize('good_fni',
        (list('abc'), tuple('abc'), set('abc'), np.array(list('abc')))
    )
    def test_accept_good_fni(self, good_fni):
        _val_ign_cols_hab_callable(
            [0, 1],
            'handle_as_bool',
            3,
            good_fni
        )


    @pytest.mark.parametrize('fni',
        (list('abc'), tuple('abc'), set('abc'), np.array(list('abc')))
    )
    @pytest.mark.parametrize('nfi', (2, 3, 4))
    def test_len_fni_against_n_features_in(self, fni, nfi):

        if  len(fni) == nfi:
            _val_ign_cols_hab_callable(
                [0, 1],
                'handle_as_bool',
                nfi,
                fni
            )
        else:
            with pytest.raises(ValueError):
                _val_ign_cols_hab_callable(
                    [0, 1],
                    'handle_as_bool',
                    nfi,
                    fni
                )
    # END validate feature_names_in -- -- -- -- -- -- -- -- -- -- -- -- -- --

    # END helper object validation ** * ** * ** * ** * ** * ** * ** * ** * **


    @pytest.mark.parametrize('junk_fxn_output',
        (-2.7, -1, 0, 1, 2.7, True, False, None, {'a':1}, lambda x: x)
    )
    def test_rejects_junk_fxn_output(self, junk_fxn_output):
        with pytest.raises(TypeError):
            _val_ign_cols_hab_callable(
                junk_fxn_output,
                'ignore_columns',
                13,
                np.array(list('abcdefghijklm'))
            )


    @pytest.mark.parametrize('bad_fxn_output',
        ((True, False), [[1,2,3], [4,5,6]], ['a', 1, 'b', 2])
    )
    def test_rejects_bad_fxn_output(self, bad_fxn_output):
        with pytest.raises(TypeError):
            _val_ign_cols_hab_callable(
                bad_fxn_output,
                'handle_as_bool',
                13,
                np.array(list('abcdefghijklm'))
            )


    @pytest.mark.parametrize('good_fxn_output',
        ([1, 2, 3], np.array(list('abc')), {0, 2, 3}, ('d', 'e', 'f'))
    )
    def test_accepts_good_fxn_output(self, good_fxn_output):

        # 1D list-like, all int or all str

        out = _val_ign_cols_hab_callable(
            good_fxn_output,
            'ignore_columns',
            13,
            np.array(list('abcdefghijklm'))
        )

        assert out is None


    @pytest.mark.parametrize('empty_fxn_output', ([], np.array([])))
    def test_passes_empty(self, empty_fxn_output):
        out = _val_ign_cols_hab_callable(
            empty_fxn_output,
            'handle_as_bool',
            12,
            None
        )

        assert out is None


    def test_rejects_str_output_no_fni(self):
        with pytest.raises(ValueError):
            _val_ign_cols_hab_callable(
                list('abc'),
                'handle_as_bool',
                12,
                None
            )


    @pytest.mark.parametrize('bad_fn', ('x', 'y', 'z'))
    def test_rejects_str_output_not_in_fni(self, bad_fn):
        with pytest.raises(TypeError):
            _val_ign_cols_hab_callable(
                bad_fn,
                'ignore_columns',
                6,
                np.array(list('abcdef'))
            )


    @pytest.mark.parametrize('bad_idxs',
        ([99, 100, 101], (-38, -33, -27), {-19, 25})
    )
    def test_rejects_index_out_of_range(self, bad_idxs):
        with pytest.raises(ValueError):
            _val_ign_cols_hab_callable(
                bad_idxs,
                'handle_as_bool',
                6,
                np.array(list('abcdef'))
            )


    @pytest.mark.parametrize('_fxn_output', (list('abc'), [-2, -1, 0], ['c', 'd']))
    @pytest.mark.parametrize('_name', ('ignore_columns', 'handle_as_bool'))
    @pytest.mark.parametrize('_feature_names_in', (None, np.array(list('abcdefgh'))))
    def test_rejects_index_out_of_range(self, _fxn_output, _name, _feature_names_in):

        # the only thing that should fail is passing str output w/o feature_names_in
        # index output works with or without feature_names_in

        if all(map(isinstance, _fxn_output, (str for _ in _fxn_output))) \
                and _feature_names_in is None:
            with pytest.raises(ValueError):
                _val_ign_cols_hab_callable(
                    _fxn_output,
                    _name,
                    8,
                    _feature_names_in
                )
        else:
            out = _val_ign_cols_hab_callable(
                _fxn_output,
                _name,
                8,
                _feature_names_in
            )

            assert out is None














