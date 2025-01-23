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
    #     _handle_as_bool: Union[InternalHandleAsBoolType, None],
    #     _ignore_columns: Union[InternalIgnoreColumnsType, None],
    #     _original_dtypes: OriginalDtypesType
    # ) -> InternalHandleAsBoolType:


    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    # handle_as_bool -- -- -- -- -- -- -- -- -- --
    @pytest.mark.parametrize('junk_hab',
        (-2.7, -1, 0, 1, 2.7, True, 'junk', {'a':1}, lambda x: x)
    )
    def test_rejects_junk_handle_as_bool(self, junk_hab):

        with pytest.raises(TypeError):
            _val_handle_as_bool_v_dtypes(
                junk_hab,
                _ignore_columns=[],
                _original_dtypes=['bin_int', 'int', 'float', 'obj']
            )


    @pytest.mark.parametrize('bad_hab',
        (list('abcd'), tuple('abcd'), set('abcd'), np.random.randint(0, 10, (4,4)))
    )
    def test_rejects_bad_handle_as_bool(self, bad_hab):

        with pytest.raises(TypeError):
            _val_handle_as_bool_v_dtypes(
                bad_hab,
                _ignore_columns=[],
                _original_dtypes=['bin_int', 'int', 'float', 'obj']
            )


    @pytest.mark.parametrize('good_hab', ([0,1,3], (-4,-3,-1), {2,3}, None))
    def test_accepts_listlike_of_ints(self, good_hab):

        out = _val_handle_as_bool_v_dtypes(
            good_hab,
            _ignore_columns=[],
            _original_dtypes=['bin_int', 'int', 'float', 'int']
        )

        assert out is None
    # END handle_as_bool -- -- -- -- -- -- -- -- -- --

    # ignore_columns -- -- -- -- -- -- -- -- -- --
    @pytest.mark.parametrize('junk_ic',
        (-2.7, -1, 0, 1, 2.7, True, 'junk', {'a':1}, lambda x: x)
    )
    def test_rejects_junk_ignore_columns(self, junk_ic):

        with pytest.raises(TypeError):
            _val_handle_as_bool_v_dtypes(
                [],
                _ignore_columns=junk_ic,
                _original_dtypes=['bin_int', 'int', 'float', 'obj']
            )


    @pytest.mark.parametrize('bad_ic',
        (list('abcd'), tuple('abcd'), set('abcd'), np.random.randint(0, 10, (4,4)))
    )
    def test_rejects_bad_ignore_columns(self, bad_ic):

        with pytest.raises(TypeError):
            _val_handle_as_bool_v_dtypes(
                [],
                _ignore_columns=bad_ic,
                _original_dtypes=['bin_int', 'int', 'float', 'obj']
            )


    @pytest.mark.parametrize('good_ic', ([0,1,3], (-4,-3,-1), {2,3}, None))
    def test_accepts_listlike_of_ints(self, good_ic):

        out = _val_handle_as_bool_v_dtypes(
            [],
            _ignore_columns=[],
            _original_dtypes=np.array(['bin_int', 'int', 'float', 'int'])
        )

        assert isinstance(out, np.ndarray)
        assert len(out) == 0
    # END ignore_columns -- -- -- -- -- -- -- -- -- --

    # original_dtypes -- -- -- -- -- -- -- -- -- --
    @pytest.mark.parametrize('junk_ogdtype',
        (-2.7, -1, 0, 1, 2.7, True, None, 'junk', [0,1], {'a':1}, lambda x: x)
    )
    def test_rejects_junk_ogdtype(self, junk_ogdtype):

        with pytest.raises(TypeError):
            _val_handle_as_bool_v_dtypes(
                [0, 2],
                [1, 3],
                junk_ogdtype
            )


    @pytest.mark.parametrize('bad_ogdtype',
        (
            list('abcd'), ['eat', 'more', 'ckikn'], set('abcd'),
            np.random.choice(list('abc'), (3,3)),
            ['bin_int', 'obj', 'int'], np.array(['INT', 'OBJ', 'INT'])
        )
    )
    def test_rejects_bad_og_dtype(self, bad_ogdtype):

        with pytest.raises((TypeError, ValueError)):
            _val_handle_as_bool_v_dtypes(
                [-1, -2],
                [-3],
                bad_ogdtype
            )


    @pytest.mark.parametrize('good_ogdtype',
        (['bin_int', 'obj', 'int'], ('int', 'obj', 'int'))
    )
    def test_accepts_good_og_dtype(self, good_ogdtype):

        out = _val_handle_as_bool_v_dtypes(
            [0, 2],
            [1],
            np.array(list(good_ogdtype))
        )

        assert isinstance(out, np.ndarray)
        assert np.array_equal(out, [0, 2])
    # END original_dtypes -- -- -- -- -- -- -- -- -- --


    # joint -- -- -- -- -- -- -- -- -- -- -- -- -- --
    @pytest.mark.parametrize('bad_hab',
        ([-10, 5, 0], (0, 2, 100), {-100, -2, -1})
    )
    def test_rejects_hab_out_of_bounds(self, bad_hab):

        with pytest.raises(ValueError):
            _val_handle_as_bool_v_dtypes(
                bad_hab,
                [],
                np.array(['int', 'int', 'bin_int', 'float', 'float'])
            )


    @pytest.mark.parametrize('bad_ic',
        ([-10, 5, 0], (0, 2, 100), {-100, -2, -1})
    )
    def test_rejects_ic_out_of_bounds(self, bad_ic):

        with pytest.raises(ValueError):
            _val_handle_as_bool_v_dtypes(
                [3, 4],
                bad_ic,
                np.array(['int', 'int', 'bin_int', 'float', 'float'])
            )
    # END joint -- -- -- -- -- -- -- -- -- -- -- -- -- --

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    @pytest.mark.parametrize('hab', ([], None))
    def test_accepts_hab_empty_or_None(self, hab):

        _original_dtypes = np.array(['int', 'int', 'obj'])

        out = _val_handle_as_bool_v_dtypes(
            hab,
            [],
            _original_dtypes
        )

        if isinstance(hab, type(None)):
            assert out is None
        else:
            assert isinstance(out, np.ndarray)
            assert np.array_equal(out, hab)


    def test_rejects_hab_on_obj(self):

        with pytest.raises(ValueError):
            _val_handle_as_bool_v_dtypes(
                [-2, -1],
                [],
                np.array(['int', 'int'] + ['obj' for _ in range(3)])
            )


    def test_accepts_ignored_hab_on_obj(self):

        out = _val_handle_as_bool_v_dtypes(
            [-4, -2, -1],
            [-3, -1],
            np.array(['int', 'int', 'int', 'obj'])
        )

        assert isinstance(out, np.ndarray)
        # -1 is ignored and hab, so that falls out of returned hab
        assert np.array_equal(out, [-4, -2])


    def test_accept_hab_on_numeric(self):

        out = _val_handle_as_bool_v_dtypes(
            [0, 1, 2],
            None,
            np.array(['int', 'float', 'bin_int', 'obj'])
        )

        assert isinstance(out, np.ndarray)
        assert np.array_equal(out, [0, 1, 2])

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        out = _val_handle_as_bool_v_dtypes(
            [-4, -3, -2],
            None,
            np.array(['int', 'float', 'bin_int', 'obj'])
        )

        assert isinstance(out, np.ndarray)
        assert np.array_equal(out, [-4, -3, -2])





