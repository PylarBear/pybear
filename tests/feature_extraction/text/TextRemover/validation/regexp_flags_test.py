# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import re

import numpy as np

from pybear.feature_extraction.text._TextRemover._validation._regexp_flags \
    import _val_regexp_flags



class TestValRexExpFlags:


    # must be None, numbers.Integral, or
    # list[Union[numbers.Integral, None, Literal[False]]]


    @pytest.mark.parametrize('junk_single_flags',
        (-2.7, 2.7, True, False, 'garbage', {'A': 1}, lambda x: x)
    )
    def test_rejects_junk_single_flags(self, junk_single_flags):

        with pytest.raises(TypeError):
            _val_regexp_flags(junk_single_flags, list('ab'))


    @pytest.mark.parametrize('junk_seq_flags',
        (list((True, False)), list('ab'), set(list((1, 2))), tuple(list((1, 2))))
    )
    def test_rejects_junk_seq_flags(self, junk_seq_flags):

        with pytest.raises(TypeError):
            _val_regexp_flags(junk_seq_flags, list('ab'))


    def test_rejects_bad_seq_flags(self):

        # too long
        with pytest.raises(ValueError):
            _val_regexp_flags(
                np.random.randint(0,100, (6,)).tolist(),
                list('abcde')
            )

        # too short
        with pytest.raises(ValueError):
            _val_regexp_flags(
                np.random.randint(0,100, (4,)).tolist(),
                list('abcde')
            )


    def test_accepts_single_None_single_int(self):

        assert _val_regexp_flags(None, list('ab')) is None

        assert _val_regexp_flags(-20, list('ab')) is None

        assert _val_regexp_flags(10_000, list('ab')) is None

        assert _val_regexp_flags(0, list('ab')) is None

        assert _val_regexp_flags(re.I | re.X, list('ab')) is None


    def test_accepts_list_of_None_False_int(self):


        for trial in range(20):

            _flags = np.random.choice(
                [int(np.random.randint(-1000, 1000)), re.I | re.X, False, None],
                (5, ),
                replace=True
            ).tolist()

            _val_regexp_flags(
                _flags,
                _X=list('abcde')
            )















