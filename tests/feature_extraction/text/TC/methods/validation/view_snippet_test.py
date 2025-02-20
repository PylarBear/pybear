# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.feature_extraction.text._TC._methods._validation._view_snippet \
    import _view_snippet_validation



class TestViewSnippetValidation:



    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @pytest.mark.parametrize('junk_vector',
        (-2.7, -1, 0, 1, 2.7, True, None, 'junk', [0,1], (1,), {1,2}, {'a':1},
         lambda x: x)
    )
    def test_rejects_junk_vector(self, junk_vector):

        with pytest.raises(TypeError):
            _view_snippet_validation(junk_vector, 0, 9)


    @pytest.mark.parametrize('junk_idx',
        (-2.7, 2.7, True, False, None, 'junk', [0,1], (1,), {1,2}, {'a':1},
         lambda x: x)
    )
    def test_rejects_junk_idx(self, junk_idx):

        with pytest.raises(TypeError):
            _view_snippet_validation(list('abcde'), junk_idx, 9)


    @pytest.mark.parametrize('bad_idx', (-1, 10, 250000))
    def test_rejects_bad_idx(self, bad_idx):

        with pytest.raises(ValueError):
            _view_snippet_validation(list('abcde'), bad_idx, 9)


    @pytest.mark.parametrize('junk_span',
        (-2.7, 2.7, True, False, None, 'junk', [0,1], (1,), {1,2}, {'a':1},
         lambda x: x)
    )
    def test_rejects_junk_span(self, junk_span):

        with pytest.raises(TypeError):
            _view_snippet_validation(list('abcde'), 0, span=junk_span)


    @pytest.mark.parametrize('bad_span', (-1, 0, 1, 2))
    def test_rejects_bad_span(self, bad_span):

        with pytest.raises(ValueError):
            _view_snippet_validation(list('abcde'), 0, span=bad_span)

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **




