# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest
import numpy as np
from pybear.feature_extraction.text._TextCleaner._lex_lookup. \
    _view_snippet import _view_snippet


class TestViewSnippet:

    @pytest.mark.parametrize('junk_VECTOR',
        (0, 3.14, True, None, min, {'a':1}, 'junk', lambda x: x)
    )
    def test_rejects_VECTOR_non_list_like(self, junk_VECTOR):
        with pytest.raises(ValueError):
            _view_snippet(junk_VECTOR, 1)

    @pytest.mark.parametrize('bad_value',
        (-1, 3.14, min, True, None, [1,2], {1,2}, {'a':1}, lambda x: x)
    )
    def test_rejects_bad_VECTOR(self, bad_value):

        with pytest.raises(ValueError):
            _view_snippet([bad_value, 'GOOD', 'WORDS'], 1)


    def test_rejects_bad_shape_VECTOR(self):

        with pytest.raises(ValueError):
            _view_snippet([['GOOD', 'WORD']], 1)

        with pytest.raises(ValueError):
            _view_snippet(np.array(['GOOD', 'WORD']).reshape((1,-1)), 1)


    @pytest.mark.parametrize('bad_idx',
        (-1, 3.14, 'a', [0,1], {0,1}, (0,1), {'a':1}, None, True, lambda x: x)
    )
    def test_bad_idx(self, bad_idx):

        with pytest.raises(ValueError):
            _view_snippet(['a','bunch', 'of', 'good', 'words'], bad_idx)

    @pytest.mark.parametrize('bad_span',
        (0, -1, 3.14, 'a', [0, 1], {0, 1}, (0, 1), {'a': 1}, None, True, lambda x: x)
    )
    def test_bad_span(self, bad_span):

        with pytest.raises(ValueError):
            _view_snippet(['a', 'bunch', 'of', 'good', 'words'], 1, span=bad_span)


    @staticmethod
    @pytest.fixture
    def good_vector():
        return [
            'When', 'in', 'the', 'Course', 'of', 'human', 'events', 'it',
            'becomes', 'necessary',  'for', 'one', 'people', 'to', 'dissolve',
            'the', 'political', 'bands', 'which', 'have',  'connected', 'them',
            'with',  'another', 'and',  'to',  'assume',  'among',  'the',
            'powers',  'of',  'the',  'earth', 'the',  'separate',  'and',
            'equal',  'station',  'to',  'which',  'the', 'Laws', 'of',
            'Nature', 'and', 'of', 'Natures','God', 'entitle', 'them','a',
            'decent','respect','to','the','opinions','of','mankind', 'requires',
            'that', 'they', 'should', 'declare', 'the', 'causes', 'which',
            'impel', 'them', 'to', 'the', 'separation'
        ]


    @pytest.mark.parametrize('idx', (0, 30, 70))
    @pytest.mark.parametrize('span', (9,))
    def test_accuracy(self, good_vector, idx, span):

        out = _view_snippet(good_vector, idx, span=span)

        if idx == 0:
            exp = good_vector[:span]
            exp = list(map(str.lower, exp))
            exp[idx] = exp[idx].upper()
            exp = " ".join(exp)
            assert out == exp


        elif idx == 30:
            _low = int(np.floor(idx - (span - 1) / 2))
            _high = int(np.ceil(idx + (span - 1) / 2))
            exp = list(map(str.lower, good_vector))
            exp[idx] = exp[idx].upper()
            exp = exp[_low : _high]
            exp = " ".join(exp)
            assert out == exp


        elif idx == 70:
            exp = list(map(str.lower, good_vector))
            exp[idx] = exp[idx].upper()
            exp = exp[-span:]
            exp = " ".join(exp)
            assert out == exp















