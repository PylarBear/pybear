# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.utilities._column_name_mapper import column_name_mapper

import numpy as np

import pytest



class TestColumnNameMapper:


    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # feature_names_in -- -- -- -- -- -- -- -- -- -- -- -- --
    @pytest.mark.parametrize('junk_fni',
        (-2.7, -1, 0, 1, 2.7, True, 'junk', {'a':1}, lambda x: x)
    )
    def test_feature_names_in_rejects_junk(self, junk_fni):
        with pytest.raises(TypeError):
            column_name_mapper(
                None,
                junk_fni,
                positive=True
            )

    @pytest.mark.parametrize('bad_fni',
        ([1, 2, 3], [True, False], np.random.randint(0,10, (3,3)), [list('abc')], [])
    )
    def test_feature_names_in_rejects_bad(self, bad_fni):
        with pytest.raises(TypeError):
            column_name_mapper(
                None,
                bad_fni,
                positive=True
            )

    @pytest.mark.parametrize('good_fni',
        (
            list('abcd'),
            tuple('abcd'),
            set('abcd'),
            np.array(list('abcd'))
        )
    )
    def test_feature_names_in_accepts_1D_of_strings(self, good_fni):

        out = column_name_mapper(
            None,
            good_fni,
            positive=True
        )

        assert out is None
    # END feature_names_in -- -- -- -- -- -- -- -- -- -- -- -- --

    # positive -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    @pytest.mark.parametrize('junk_positive',
        (-2.7, -1, 0, 1, 2.7, 'junk', [0,1], (1,), {'a':1}, lambda x: x)
    )
    def test_positive_rejects_junk(self, junk_positive):
        with pytest.raises(TypeError):
            column_name_mapper(
                None,
                list('abcd'),
                junk_positive
            )


    @pytest.mark.parametrize('good_positive', (True, False, None))
    def test_positive_accepts_bool_or_none(self, good_positive):

        out = column_name_mapper(
            None,
            list('abcdef'),
            positive=good_positive
        )

        assert out is None
    # END positive -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    # feature_names -- -- -- -- -- -- -- -- -- -- -- -- -- --
    @pytest.mark.parametrize('junk_fn',
        (-2.7, -1, 0, 1, 2.7, True, 'junk', {'a':1}, lambda x: x)
    )
    def test_feature_names_rejects_junk(self, junk_fn):
        with pytest.raises(TypeError):
            column_name_mapper(
                junk_fn,
                set('abcde'),
                positive=True
            )


    @pytest.mark.parametrize('bad_fn',
        ([True, False], np.random.randint(0,10, (3,3)))
    )
    def test_feature_names_rejects_bad(self, bad_fn):
        with pytest.raises(TypeError):
            column_name_mapper(
                bad_fn,
                tuple('abcde'),
                positive=True
            )


    @pytest.mark.parametrize('good_fn',
        (
            list('abcd'),
            tuple('abcd'),
            set('abcd'),
            np.array(list('abcd')),
            [1,2,3],
            set((-1, -2, -3)),
            (0, 1, 2),
             np.array([12, 13, 14])
        )
    )
    def test_feature_names_accepts_good(self, good_fn):

        out = column_name_mapper(
            good_fn,
            np.array(list('abcdefghijklmnop')),
            positive=None
        )

        assert isinstance(out, np.ndarray)
    # feature_names -- -- -- -- -- -- -- -- -- -- -- -- -- --

    # joint -- -- -- -- -- -- -- -- -- -- -- -- -- --
    @pytest.mark.parametrize('bad_fn_indices',
        ([-100, -99, -98], [98, 99, 100])
    )
    def test_fn_indices_out_of_range(self, bad_fn_indices):

        with pytest.raises(ValueError):
            column_name_mapper(
                bad_fn_indices,
                tuple('abcde'),
                positive=None
            )


    def test_feature_names_as_str_no_fni(self):

        with pytest.raises(ValueError):
            column_name_mapper(
                list('abc'),
                None,
                positive=None
            )
    # END joint -- -- -- -- -- -- -- -- -- -- -- -- -- --


    # accuracy -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    @pytest.mark.parametrize('feature_names',
        (
            [-2, -1], [1, 3], [0, -1], ['c', 'd'], ('d', 'b'), {'a', 'd'},
            list('aaabbb'), list('ccbb')
        )
    )
    @pytest.mark.parametrize('feature_names_in', (list('abcdefg'), ))
    @pytest.mark.parametrize('positive', (True, False, None))
    def test_accuracy(
        self, feature_names, feature_names_in, positive
    ):

        # pytest appears to have erratic behavior when injecting a set.
        # sometimes the {'a', 'd'} set is being passed to column_name_mapper
        # as {'d', 'a'}. when this happens, skip the test:
        if list(feature_names) == ['d', 'a']:
            pytest.skip(reason=f"pytest changed the input")

        if positive not in [True, False, None]:
            raise Exception

        out = column_name_mapper(
            feature_names,
            feature_names_in,
            positive=positive
        )

        assert isinstance(out, np.ndarray)
        assert out.dtype == np.int32

        max_dim = len(feature_names_in)

        if feature_names == [-2, -1]:
            if positive is True:
                assert np.array_equiv(out, [max_dim - 2, max_dim - 1])
            elif positive is False:
                assert np.array_equiv(out, [-2, -1])
            elif positive is None:
                assert np.array_equiv(out, [-2, -1])
        elif feature_names == [1, 3]:
            if positive is True:
                assert np.array_equiv(out, [1, 3])
            elif positive is False:
                assert np.array_equiv(out, [1 - max_dim, 3 - max_dim])
            elif positive is None:
                assert np.array_equiv(out, [1, 3])
        elif feature_names == [0, -1]:
            if positive is True:
                assert np.array_equiv(out, [0, max_dim-1])
            elif positive is False:
                assert np.array_equiv(out, [-max_dim, -1])
            elif positive is None:
                assert np.array_equiv(out, [0, -1])
        elif feature_names == ['c', 'd']:
            if positive is True:
                assert np.array_equiv(out, [2, 3])
            elif positive is False:
                assert np.array_equiv(out, [-max_dim+2, -max_dim+3])
            elif positive is None:
                assert np.array_equiv(out, [2, 3])
        elif feature_names == ('d', 'b'):
            if positive is True:
                assert np.array_equiv(out, [3, 1])
            elif positive is False:
                assert np.array_equiv(out, [-max_dim+3, -max_dim+1])
            elif positive is None:
                assert np.array_equiv(out, [3, 1])
        elif feature_names == {'a', 'd'}:
            # set sorts them
            if positive is True:
                assert np.array_equiv(out, [0, 3])
            elif positive is False:
                assert np.array_equiv(out, [-max_dim, -max_dim+3])
            elif positive is None:
                assert np.array_equiv(out, [0, 3])
        elif feature_names == list('aaabbb'):
            if positive is True:
                assert np.array_equiv(out, [0, 0, 0, 1, 1, 1])
            elif positive is False:
                assert np.array_equiv(
                    out,
                    [-max_dim, -max_dim, -max_dim, -max_dim+1, -max_dim+1, -max_dim+1],
                )
            elif positive is None:
                assert np.array_equiv(out, [0, 0, 0, 1, 1, 1])
        elif feature_names == list('ccbb'):
            if positive is True:
                assert np.array_equiv(out, [2, 2, 1, 1])
            elif positive is False:
                assert np.array_equiv(
                    out,
                    [-max_dim+2, -max_dim+2, -max_dim+1, -max_dim+1]
                )
            elif positive is None:
                assert np.array_equiv(out, [2, 2, 1, 1])
        else:
            raise Exception
















