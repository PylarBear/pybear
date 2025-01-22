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


    #  def _val_handle_as_bool_v_dtypes(
    #     _handle_as_bool: Iterable[numbers.Integral],
    #     _original_dtypes: OriginalDtypesDtype
    # ) -> None:


    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    # handle_as_bool -- -- -- -- -- -- -- -- -- --
    @pytest.mark.parametrize('junk_hab',
        (-2.7, -1, 0, 1, 2.7, True, 'junk', {'a':1}, lambda x: x)
    )
    def test_rejects_junk_handle_as_bool(self, junk_hab):

        with pytest.raises(TypeError):
            _val_handle_as_bool_v_dtypes(
                junk_hab,
                ['bin_int', 'int', 'float', 'obj']
            )


    @pytest.mark.parametrize('bad_hab',
        (list('abcd'), tuple('abcd'), np.random.randint(0, 10, (4,4)))
    )
    def test_rejects_bad_handle_as_bool(self, bad_hab):

        with pytest.raises(TypeError):
            _val_handle_as_bool_v_dtypes(
                bad_hab,
                ['bin_int', 'int', 'float', 'obj']
            )


    @pytest.mark.parametrize('good_hab', ([0,1,3], (-4,-3,-1), {2,3}, None))
    def test_accepts_listlike_of_ints(self, good_hab):

        out = _val_handle_as_bool_v_dtypes(
            good_hab,
            ['bin_int', 'int', 'float', 'int']
        )

        assert out is None
    # END handle_as_bool -- -- -- -- -- -- -- -- -- --

    # original_dtypes -- -- -- -- -- -- -- -- -- --
    @pytest.mark.parametrize('junk_ogdtype',
        (-2.7, -1, 0, 1, 2.7, True, None, 'junk', [0,1], {'a':1}, lambda x: x)
    )
    def test_rejects_junk_ogdtype(self, junk_ogdtype):

        with pytest.raises(TypeError):
            _val_handle_as_bool_v_dtypes(
                [0, 2],
                junk_ogdtype
            )


    @pytest.mark.parametrize('bad_ogdtype',
        (list('abcd'), ['eat', 'more', 'ckikn'],
         np.random.choice(list('abc'), (3,3)))
    )
    def test_rejects_bad_og_dtype(self, bad_ogdtype):

        with pytest.raises((TypeError, ValueError)):
            _val_handle_as_bool_v_dtypes(
                [-1, -2],
                bad_ogdtype
            )


    @pytest.mark.parametrize('good_ogdtype',
        (['bin_int', 'obj', 'int'], ('INT', 'OBJ', 'INT'))
    )
    def test_accepts_good_og_dtype(self, good_ogdtype):

        out = _val_handle_as_bool_v_dtypes(
            [0, 2],
            good_ogdtype
        )

        assert out is None
    # END original_dtypes -- -- -- -- -- -- -- -- -- --


    # joint -- -- -- -- -- -- -- -- -- -- -- -- -- --
    @pytest.mark.parametrize('bad_hab',
        ([-10, 5, 0], (0, 2, 100), {-100, -2, -1})
    )
    def test_rejects_hab_out_of_bounds(self, bad_hab):

        with pytest.raises(ValueError):
            _val_handle_as_bool_v_dtypes(
                bad_hab,
                ['int', 'int', 'bin_int', 'float', 'float']
            )

    # END joint -- -- -- -- -- -- -- -- -- -- -- -- -- --

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    def test_accept_empty_hab(self):
        out = _val_handle_as_bool_v_dtypes(
            [],
            ['int', 'int', 'obj']
        )

        assert out is None


    def test_rejects_hab_on_obj(self):

        with pytest.raises(ValueError):
            _val_handle_as_bool_v_dtypes(
                [-2, -1],
                ['int', 'int'] + ['obj' for _ in range(3)]
            )


    def test_accept_hab_on_numeric(self):

        out = _val_handle_as_bool_v_dtypes(
            [0, 1, 2],
            ['int', 'float', 'bin_int', 'obj']
        )

        assert out is None

        out = _val_handle_as_bool_v_dtypes(
            [-4, -3, -2],
            ['int', 'float', 'bin_int', 'obj']
        )

        assert out is None





