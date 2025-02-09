# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.feature_extraction.text._TextStatistics._partial_fit._merge_uniques \
    import _merge_uniques



class TestBuildUniques:


    @pytest.mark.parametrize('junk_current_uniques',
        (-2.7, -1, 0, 1, 2.7, True, False, None, 'trash', {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_current_uniques(self, junk_current_uniques):

        with pytest.raises(TypeError):
            _merge_uniques(
                junk_current_uniques,
                list('abcde')
            )


    @pytest.mark.parametrize('junk_uniques',
        (-2.7, -1, 0, 1, 2.7, None, 'trash', {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_uniques(self, junk_uniques):
        with pytest.raises(TypeError):
            _merge_uniques(
                list('abcdefghijkl'),
                junk_uniques
            )


    def test_accuracy(self):

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        out = _merge_uniques(
            list('abcdef'),
            list('abcdef')
        )

        assert np.array_equal(
            sorted(list(out)),
            list('abcdef')
        )

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        out = _merge_uniques(
            list('aBcDeF'),
            list('aBcDeF')
        )

        assert np.array_equal(
            sorted(list(out)),
            list('BDFace')
        )

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        out = _merge_uniques(
            ['I', 'am', 'Sam'],
            ['That', 'Sam-I-am', 'Sam', 'I', 'am']
        )

        assert np.array_equal(
            sorted(list(out)),
            ['I', 'Sam', 'Sam-I-am', 'That', 'am']
        )

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

        out = _merge_uniques(
            ['I', 'AM', 'SAM'],
            ['SAM', 'I', 'AM', 'THAT', 'SAM-I-AM']
        )

        assert np.array_equal(
            sorted(list(out)),
            ['AM', 'I', 'SAM', 'SAM-I-AM', 'THAT']
        )
















