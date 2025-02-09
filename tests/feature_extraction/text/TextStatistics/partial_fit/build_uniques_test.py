# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.feature_extraction.text._TextStatistics._partial_fit._build_uniques \
    import _build_uniques



class TestBuildUniques:

    @pytest.mark.parametrize('junk_words',
        (-2.7, -1, 0, 1, 2.7, True, False, None, [0,1], (1,), {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_words(self, junk_words):

        with pytest.raises(Exception):
            _build_uniques(
                junk_words,
                case_sensitive=False
            )

    @pytest.mark.parametrize('junk_case_sensitive',
        (-2.7, -1, 0, 1, 2.7, None, [0, 1], (1,), {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_case_sensitive(self, junk_case_sensitive):
        with pytest.raises(Exception):
            _build_uniques(
                list('abcdefghijkl'),
                case_sensitive=junk_case_sensitive
            )


    @pytest.mark.parametrize('case_sensitive', (True, False))
    def test_accuracy(self, case_sensitive):

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        WORDS1 = list('abcdef')

        out = _build_uniques(
            WORDS1,
            case_sensitive=case_sensitive
        )

        if case_sensitive:
            assert np.array_equal(
                sorted(list(out)),
                list('abcdef')
            )
        elif not case_sensitive:
            assert np.array_equal(
                sorted(list(out)),
                list('ABCDEF')
            )

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        WORDS2 = list('aBcDeF')

        out = _build_uniques(
            WORDS2,
            case_sensitive=case_sensitive
        )

        if case_sensitive:
            assert np.array_equal(
                sorted(list(out)),
                list('BDFace')
            )
        elif not case_sensitive:
            assert np.array_equal(
                sorted(list(out)),
                list('ABCDEF')
            )

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        WORDS3 = ['I', 'am', 'Sam', 'Sam', 'I', 'am', 'That', 'Sam-I-am']

        out = _build_uniques(
            WORDS3,
            case_sensitive=case_sensitive
        )

        if case_sensitive:
            assert np.array_equal(
                sorted(list(out)),
                ['I', 'Sam', 'Sam-I-am', 'That', 'am']
            )
        elif not case_sensitive:
            assert np.array_equal(
                sorted(list(out)),
                ['AM', 'I', 'SAM', 'SAM-I-AM', 'THAT']
            )

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        WORDS4 = ['I', 'AM', 'SAM', 'SAM', 'I', 'AM', 'THAT', 'SAM-I-AM']

        out = _build_uniques(
            WORDS4,
            case_sensitive=case_sensitive
        )

        if case_sensitive:
            assert np.array_equal(
                sorted(list(out)),
                ['AM', 'I', 'SAM', 'SAM-I-AM', 'THAT']
            )
        elif not case_sensitive:
            assert np.array_equal(
                sorted(list(out)),
                ['AM', 'I', 'SAM', 'SAM-I-AM', 'THAT']
            )
















