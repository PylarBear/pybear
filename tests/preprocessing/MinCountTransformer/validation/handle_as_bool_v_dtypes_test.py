# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.MinCountTransformer._validation._handle_as_bool_v_dtypes \
    import _val_handle_as_bool_v_dtypes

import numpy as np

import pytest



class TestHandleAsBoolVDtypes:


    # def _val_handle_as_bool_v_dtypes(
    #     _handle_as_bool: Union[Iterable[numbers.Integral], None],
    #     _ignore_columns: Union[Iterable[numbers.Integral], None],
    #     _original_dtypes: Iterable[numbers.Integral]
    # ) -> InternalHandleAsBoolType:


    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # raise -- -- -- -- -- -- -- -- -- -- --
    @pytest.mark.parametrize('junk_raise',
        (-2.7, -1, 0, 1, 2.7, None, 'junk', [0,1], {'a':1}, lambda x: x)
    )
    def test_rejects_junk_raise(self, junk_raise):

        with pytest.raises(TypeError):
            _val_handle_as_bool_v_dtypes(
                [0, 2],
                [1, 3],
                ['obj', 'float', 'obj', 'bin_int'],
                _raise=junk_raise
            )


    @pytest.mark.parametrize('bool_raise', (True, False))
    def test_accepts_bool_raise(self, bool_raise):

        _val_handle_as_bool_v_dtypes(
            [-1, -2],
            [-3],
            ['obj', 'int', 'float'],
            _raise=bool_raise
        )
    # END raise -- -- -- -- -- -- -- -- -- --

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    @pytest.mark.parametrize('hab', ([], None))
    def test_accepts_hab_empty_or_None(self, hab):

        out = _val_handle_as_bool_v_dtypes(
            hab,
            [],
            np.array(['int', 'int', 'obj'])
        )

        assert out is None


    def test_warns_excepts_hab_on_obj(self):

        with pytest.raises(ValueError):
            _val_handle_as_bool_v_dtypes(
                [-2, -1],
                [],
                ['int', 'int'] + ['obj' for _ in range(3)],
                _raise=True
            )

        with pytest.warns():
            _val_handle_as_bool_v_dtypes(
                [-2, -1],
                [],
                ['int', 'int'] + ['obj' for _ in range(3)],
                _raise=False
            )


    def test_accepts_ignored_hab_on_obj(self):

        out = _val_handle_as_bool_v_dtypes(
            [-4, -2, -1],
            [-3, -1],
            tuple(['int', 'int', 'int', 'obj'])
        )

        assert out is None


    def test_accept_hab_on_numeric(self):

        out = _val_handle_as_bool_v_dtypes(
            [0, 1, 2],
            None,
            set(('int', 'float', 'bin_int', 'obj'))
        )

        assert out is None

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        out = _val_handle_as_bool_v_dtypes(
            [-4, -3, -2],
            None,
            np.array(['int', 'float', 'bin_int', 'obj'])
        )

        assert out is None





