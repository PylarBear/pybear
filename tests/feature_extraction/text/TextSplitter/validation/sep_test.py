# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import uuid

import numpy as np

from pybear.feature_extraction.text._TextSplitter._validation._sep import \
    _val_sep




class TestValSep:


    @pytest.mark.parametrize('junk_sep',
        (-2.7, -1, 0, 1, 2.7, True, False, {'A': 1}, lambda x: x)
    )
    def test_rejects_junk_sep_as_single(self, junk_sep):
        # could be None, str, set[str], list[of the 3]
        with pytest.raises(TypeError):
            _val_sep(junk_sep, list('abcde'))


    @pytest.mark.parametrize('good_single_sep', (None, ', ', {' ', ', ', '. '}))
    def test_accepts_good_single(self, good_single_sep):
        # could be None, str, set[str]
        _val_sep(good_single_sep, list('abcde'))


    @pytest.mark.parametrize('junk_container', (tuple, list, np.ndarray))
    @pytest.mark.parametrize('junk_seps', ([True, False], (0, 1), ('a', 'b')))
    def test_rejects_junk_sep_as_seq(self, junk_container, junk_seps):
        # could be None, str, set[str], or list[of the 3]

        # the only one that should be good is list / ('a', 'b')
        if junk_container is list \
                and all(map(isinstance, junk_seps, (str for _ in junk_seps))):
            pytest.skip(reason=f"should pass")

        if junk_container is np.ndarray:
            junk_sep = np.array(junk_seps)
            assert isinstance(junk_sep, np.ndarray)
        else:
            junk_sep = junk_container(junk_seps)
            assert isinstance(junk_sep, junk_container)

        with pytest.raises(TypeError):
            _val_sep(junk_sep, list('ab'))


    def test_rejects_bad_sep_as_seq(self):

        # too long
        with pytest.raises(ValueError):
            _val_sep([str(uuid.uuid4())[:4] for _ in range(6)], list('abcde'))


        # too short
        with pytest.raises(ValueError):
            _val_sep([str(uuid.uuid4())[:4] for _ in range(4)], list('abcde'))


    @pytest.mark.parametrize('innards', (None, ' ', set((' ', ', ', '. '))))
    def test_accepts_good_sequence(self, innards):
        # list[of the 3]

        _sep = [innards for _ in range(5)]

        assert _val_sep(_sep, list('abcde')) is None




