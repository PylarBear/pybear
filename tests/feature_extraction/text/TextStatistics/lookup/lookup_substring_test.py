# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.feature_extraction.text._TextStatistics._lookup._lookup_substring \
    import _lookup_substring



class TestLookupSubstring:


    # def _lookup_substring(
    #     char_seq: str,
    #     uniques: Sequence[str],
    #     case_sensitive: Optional[bool] = True
    # ) -> Sequence[str]:


    @staticmethod
    @pytest.fixture(scope='function')
    def uniques():
        return [
            'Do you like ',
            'green eggs and ham?',
            'I do not like them, Sam-I-am.',
            'I do not like',
            'green eggs and ham.'
        ]

    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    # char_seq -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    @pytest.mark.parametrize('junk_char_seq',
        (-2.7, -1, 0, 1, 2.7, True, None, [0,1], ('a', ), {'a': 1}, lambda x: x)
    )
    def test_blocks_non_str_char_seq(self, junk_char_seq, uniques):

        with pytest.raises(TypeError):
            _lookup_substring(junk_char_seq, uniques, True)


    @pytest.mark.parametrize('char_seq', ('green', 'eggs', 'and', 'ham'))
    def test_accepts_str_char_seq(self, char_seq, uniques):

        _lookup_substring('look me up', uniques, case_sensitive=True)
    # END char_seq -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    # uniques -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    @pytest.mark.parametrize('junk_uniques',
        (-2.7, -1, 0, 1, 2.7, None, 'trash', [0,1], (1, ), {'a': 1}, lambda x: x)
    )
    def test_blocks_non_sequence_str_uniques(self, junk_uniques):

        with pytest.raises(TypeError):
            _lookup_substring('look me up', junk_uniques, case_sensitive=False)


    def test_blocks_empty_uniques(self):
        with pytest.raises(TypeError):
            _lookup_substring('look me up', [], case_sensitive=False)


    @pytest.mark.parametrize('container', (set, tuple, list, np.ndarray))
    def test_accepts_sequence_str_uniques(self, container, uniques):

        if container is np.ndarray:
            uniques = np.array(uniques)
        else:
            uniques = container(uniques)

        assert isinstance(uniques, container)

        _lookup_substring('look me up', uniques, case_sensitive=False)
    # END uniques -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    # case_sensitive -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    @pytest.mark.parametrize('junk_case_sensitive',
        (-2.7, -1, 0, 1, 2.7, None, 'trash', [0,1], ('a', ), {'a': 1}, lambda x: x)
    )
    def test_blocks_non_bool_case_sensitive(self, junk_case_sensitive, uniques):

        with pytest.raises(TypeError):
            _lookup_substring('look me up', uniques, junk_case_sensitive)


    @pytest.mark.parametrize('case_sensitive', (True, False))
    def test_accepts_bool_case_sensitive(self, case_sensitive, uniques):

        _lookup_substring('look me up', uniques, case_sensitive)
    # END case_sensitive -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    @pytest.mark.parametrize('case_sensitive', (True, False))
    @pytest.mark.parametrize('test_string',
        ('gree', '-I-', '-i-', 'HAM', 'n e', 'o not l', 'O NOT L')
    )
    @pytest.mark.parametrize('container', (tuple, list, np.ndarray))
    def test_accuracy(
        self, case_sensitive, test_string, container, uniques
    ):

        # dont test sets here, it messes up the order. we did prove above
        # that sets are accepted, though.

        # 'Do you like ',
        # 'green eggs and ham?',
        # 'I do not like them, Sam-I-am.',
        # 'I do not like',
        # 'green eggs and ham.'

        if container is np.ndarray:
            uniques = np.array(uniques)
        else:
            uniques = container(uniques)

        out = _lookup_substring(
            test_string,
            uniques,
            case_sensitive=case_sensitive
        )

        if case_sensitive:

            if test_string == 'gree':
                assert np.array_equal(
                    out,
                    np.array(list(uniques))[[1, 4]]
                )
            elif test_string == '-I-':
                assert np.array_equal(
                    out,
                    np.array(list(uniques))[[2]]
                )
            elif test_string == '-i-':
                assert np.array_equal(out, [])
            elif test_string == 'HAM':
                assert np.array_equal(out, [])
            elif test_string == 'n e':
                assert np.array_equal(
                    out,
                    np.array(list(uniques))[[1, 4]]
                )
            elif test_string == 'o not l':
                assert np.array_equal(
                    out,
                    np.array(list(uniques))[[2, 3]]
                )
            elif test_string == 'O NOT L':
                assert np.array_equal(out, [])
            else:
                raise Exception

        elif not case_sensitive:

            if test_string == 'gree':
                assert np.array_equal(
                    out,
                    np.array(list(uniques))[[1, 4]]
                )
            elif test_string == '-I-':
                assert np.array_equal(
                    out,
                    np.array(list(uniques))[[2]]
                )
            elif test_string == '-i-':
                assert np.array_equal(
                    out,
                    np.array(list(uniques))[[2]]
                )
            elif test_string == 'HAM':
                assert np.array_equal(
                    out,
                    np.array(list(uniques))[[1, 4]]
                )
            elif test_string == 'n e':
                assert np.array_equal(
                    out,
                    np.array(list(uniques))[[1, 4]]
                )
            elif test_string == 'o not l':
                assert np.array_equal(
                    out,
                    np.array(list(uniques))[[2, 3]]
                )
            elif test_string == 'O NOT L':
                assert np.array_equal(
                    out,
                    np.array(list(uniques))[[2, 3]]
                )
            else:
                raise Exception


















