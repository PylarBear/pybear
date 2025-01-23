# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.MinCountTransformer._fit._original_dtypes_merger \
    import _original_dtypes_merger

import numpy as np

import pytest



class TestOriginalDtypesMerger:


    #  def _original_dtypes_merger(
    #     _col_dtypes: OriginalDtypesType,
    #     _previous_col_dtypes: Union[OriginalDtypesType, None]
    # ) -> OriginalDtypesType:


    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    # _col_dtypes -- -- -- -- -- -- -- -- -- --
    @pytest.mark.parametrize('junk_col_dtypes',
        (-2.7, -1, 0, 1, 2.7, True, 'junk', {'a':1}, lambda x: x)
    )
    def test_rejects_junk_col_dtypes(self, junk_col_dtypes):

        with pytest.raises(TypeError):
            _original_dtypes_merger(
                junk_col_dtypes,
                ['obj', 'float', 'int', 'bin_int']
            )


    @pytest.mark.parametrize('bad_col_dtypes',
        (list('abcd'), tuple('abcd'), set('abcd'), np.array(['junk', 'trash']))
    )
    def test_rejects_bad_col_dtypes(self, bad_col_dtypes):

        with pytest.raises(ValueError):
            _original_dtypes_merger(
                bad_col_dtypes,
                ['obj', 'float', 'int', 'bin_int']
            )


    def test_accepts_good_col_dtypes(self):

        out = _original_dtypes_merger(
            ['bin_int', 'obj', 'float', 'int'],
            ['obj', 'float', 'int', 'bin_int']
        )

        assert isinstance(out, np.ndarray)
        assert all(map(isinstance, out, (str for _ in out)))
    # END _col_dtypes -- -- -- -- -- -- -- -- -- --

    # _previous_col_dtypes -- -- -- -- -- -- -- -- -- --
    @pytest.mark.parametrize('junk_previous_col_dtypes',
        (-2.7, -1, 0, 1, 2.7, True, 'junk', [0,1], {'a':1}, lambda x: x)
    )
    def test_rejects_junk_previous_col_dtypes(self, junk_previous_col_dtypes):

        with pytest.raises(TypeError):
            _original_dtypes_merger(
                ['obj', 'float', 'int', 'bin_int'],
                junk_previous_col_dtypes
            )


    @pytest.mark.parametrize('bad_previous_col_dtypes',
        (list('abcd'), tuple('abcd'), set('abcd'), np.array(['junk', 'trash']))
    )
    def test_rejects_bad_previous_col_dtypes(self, bad_previous_col_dtypes):

        with pytest.raises(ValueError):
            _original_dtypes_merger(
                ['obj', 'float', 'int', 'bin_int'],
                bad_previous_col_dtypes
            )


    def test_accepts_good_previous_col_dtypes(self):

        out = _original_dtypes_merger(
            ['bin_int', 'obj', 'float', 'int'],
            ['obj', 'float', 'int', 'bin_int']
        )

        assert isinstance(out, np.ndarray)
        assert all(map(isinstance, out, (str for _ in out)))
    # END _previous_col_dtypes -- -- -- -- -- -- -- -- -- --


    # joint -- -- -- -- -- -- -- -- -- -- -- -- -- --
    @pytest.mark.parametrize('length_mismatch',
        ((1, 10), (10, 1), (5, 6), (6, 5))
    )
    def test_rejects_different_length(self, length_mismatch):

        _allowed = ['obj', 'float', 'int', 'bin_int']

        with pytest.raises(AssertionError):
            _original_dtypes_merger(
                np.random.choice(_allowed, length_mismatch[0]),
                np.random.choice(_allowed, length_mismatch[1])
            )

    # END joint -- -- -- -- -- -- -- -- -- -- -- -- -- --

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    _dtypes = ['obj', 'float', 'int', 'bin_int']
    @pytest.mark.parametrize('_col_dtypes',
        (list(_dtypes), tuple(_dtypes), set(_dtypes), np.array(_dtypes))
    )
    def test_previous_col_dtypes_None(self, _col_dtypes):

        out = _original_dtypes_merger(
            _col_dtypes,
            None
        )

        assert isinstance(out, np.ndarray)
        assert all(map(isinstance, out, (str for _ in out)))
        assert np.array_equal(
            out,
            list(_col_dtypes)
        )

    del _dtypes


    def test_accuracy(self):

        # -- -- -- -- -- -- -- -- -- -- -- -- -- --
        out = _original_dtypes_merger(
            ['obj', 'float', 'int', 'bin_int'],
            ['obj', 'float', 'int', 'bin_int']
        )

        assert isinstance(out, np.ndarray)
        assert all(map(isinstance, out, (str for _ in out)))
        assert np.array_equal(
            out,
            ['obj', 'float', 'int', 'bin_int']
        )
        # -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # -- -- -- -- -- -- -- -- -- -- -- -- -- --
        out = _original_dtypes_merger(
            ['float', 'bin_int', 'int', 'bin_int'],
            ['obj', 'float', 'obj', 'float']
        )

        assert isinstance(out, np.ndarray)
        assert all(map(isinstance, out, (str for _ in out)))
        assert np.array_equal(
            out,
            ['obj', 'float', 'obj', 'float']
        )
        # -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # -- -- -- -- -- -- -- -- -- -- -- -- -- --
        out = _original_dtypes_merger(
            ['bin_int', 'int', 'float', 'obj'],
            ['obj', 'float', 'int', 'bin_int']
        )

        assert isinstance(out, np.ndarray)
        assert all(map(isinstance, out, (str for _ in out)))
        assert np.array_equal(
            out,
            ['obj', 'float', 'float', 'obj']
        )
        # -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # -- -- -- -- -- -- -- -- -- -- -- -- -- --
        out = _original_dtypes_merger(
            ['obj', 'float', 'obj', 'bin_int'],
            ['bin_int', 'obj', 'int', 'obj']
        )

        assert isinstance(out, np.ndarray)
        assert all(map(isinstance, out, (str for _ in out)))
        assert np.array_equal(
            out,
            ['obj', 'obj', 'obj', 'obj']
        )
        # -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # -- -- -- -- -- -- -- -- -- -- -- -- -- --
        out = _original_dtypes_merger(
            ['float', 'int', 'float', 'bin_int'],
            ['int', 'bin_int', 'bin_int', 'bin_int']
        )

        assert isinstance(out, np.ndarray)
        assert all(map(isinstance, out, (str for _ in out)))
        assert np.array_equal(
            out,
            ['float', 'int', 'float', 'bin_int']
        )
        # -- -- -- -- -- -- -- -- -- -- -- -- -- --

        # -- -- -- -- -- -- -- -- -- -- -- -- -- --
        out = _original_dtypes_merger(
            ['int', 'bin_int', 'int', 'bin_int'],
            ['bin_int', 'int', 'bin_int', 'int']
        )

        assert isinstance(out, np.ndarray)
        assert all(map(isinstance, out, (str for _ in out)))
        assert np.array_equal(
            out,
            ['int', 'int', 'int', 'int']
        )
        # -- -- -- -- -- -- -- -- -- -- -- -- -- --












