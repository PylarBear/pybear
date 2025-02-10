# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.feature_extraction.text._TextStatistics._lookup._lookup_string \
    import _lookup_string



class TestLookupSubstring:


    # def _lookup_string(
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
            _lookup_string(junk_char_seq, uniques, True)


    @pytest.mark.parametrize('char_seq', ('green', 'eggs', 'and', 'ham'))
    def test_accepts_str_char_seq(self, char_seq, uniques):

        _lookup_string('look me up', uniques, case_sensitive=True)
    # END char_seq -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    # uniques -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    @pytest.mark.parametrize('junk_uniques',
        (-2.7, -1, 0, 1, 2.7, None, 'trash', [0,1], (1, ), {'a': 1}, lambda x: x)
    )
    def test_blocks_non_sequence_str_uniques(self, junk_uniques):

        with pytest.raises(TypeError):
            _lookup_string('look me up', junk_uniques, case_sensitive=False)


    def test_blocks_empty_uniques(self):
        with pytest.raises(TypeError):
            _lookup_string('look me up', [], case_sensitive=False)


    @pytest.mark.parametrize('container', (set, tuple, list, np.ndarray))
    def test_accepts_sequence_str_uniques(self, container, uniques):

        if container is np.ndarray:
            uniques = np.array(uniques)
        else:
            uniques = container(uniques)

        assert isinstance(uniques, container)

        _lookup_string('look me up', uniques, case_sensitive=False)
    # END uniques -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    # case_sensitive -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    @pytest.mark.parametrize('junk_case_sensitive',
        (-2.7, -1, 0, 1, 2.7, None, 'trash', [0,1], ('a', ), {'a': 1}, lambda x: x)
    )
    def test_blocks_non_bool_case_sensitive(self, junk_case_sensitive, uniques):

        with pytest.raises(TypeError):
            _lookup_string('look me up', uniques, junk_case_sensitive)


    @pytest.mark.parametrize('case_sensitive', (True, False))
    def test_accepts_bool_case_sensitive(self, case_sensitive, uniques):

        _lookup_string('look me up', uniques, case_sensitive)
    # END case_sensitive -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    @pytest.mark.parametrize('case_sensitive', (True, False))
    @pytest.mark.parametrize('test_string',
        ('Do you like ', '-I-', '-i-', 'I do not like ',
         'green eggs and ham.', 'GREEN EGGS AND HAM.')
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

        out = _lookup_string(
            test_string,
            uniques,
            case_sensitive=case_sensitive
        )

        if case_sensitive:

            if test_string == 'Do you like ':
                assert out == 'Do you like '
            elif test_string == '-I-':
                assert out is None
            elif test_string == '-i-':
                assert out is None
            elif test_string == 'I do not like ':
                # notice the extra space
                assert out is None
            elif test_string == 'green eggs and ham.':
                assert out == 'green eggs and ham.'
            elif test_string == 'GREEN EGGS AND HAM.':
                assert out is None
            else:
                raise Exception

        elif not case_sensitive:

            if test_string == 'Do you like ':
                assert np.array_equal(out, ['Do you like '])
            elif test_string == '-I-':
                assert out is None
            elif test_string == '-i-':
                assert out is None
            elif test_string == 'I do not like ':
                # notice the extra space
                assert out is None
            elif test_string == 'green eggs and ham.':
                assert np.array_equal(out, ['green eggs and ham.'])
            elif test_string == 'GREEN EGGS AND HAM.':
                assert np.array_equal(out, ['green eggs and ham.'])
            else:
                raise Exception


    @pytest.mark.parametrize('test_string',
        ('Do you like ', 'GREEN eggs AND ham?', 'green eggs and ham?',
         'GREEN EGGS AND HAM?')
    )
    @pytest.mark.parametrize('container', (tuple, list, np.ndarray))
    def test_accuracy_multiple_returns(self, test_string, container):

        uniques = [
            'Do you like ',
            'green eggs and ham?',
            'GrEeN eGgS aNd HaM?',
            'I do not like',
            'GREEN EGGS AND HAM?'
        ]

        if container is np.ndarray:
            uniques = np.array(uniques)
        else:
            uniques = container(uniques)

        out = _lookup_string(
            test_string,
            uniques,
            case_sensitive=False   # <========================
        )

        exp = ['green eggs and ham?', 'GrEeN eGgS aNd HaM?', 'GREEN EGGS AND HAM?']

        if test_string == 'Do you like ':
            assert np.array_equal(out, ['Do you like '])
        elif test_string == 'GREEN eggs AND ham?':
            assert np.array_equal(set(out), set(exp))
        elif test_string == 'green eggs and ham?':
            assert np.array_equal(set(out), set(exp))
        elif test_string == 'GREEN EGGS AND HAM?':
            assert np.array_equal(set(out), set(exp))
        else:
            raise Exception















