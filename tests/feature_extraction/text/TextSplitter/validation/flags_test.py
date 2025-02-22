# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.feature_extraction.text._TextSplitter._validation._flags import \
    _val_flags




class TestValFlags:


    # must be None, numbers.Integral, or Sequence[numbers.Integral]


    @pytest.mark.parametrize('junk_flags',
        (-2.7, 2.7, True, False, 'garbage', {'A': 1}, lambda x: x)
    )
    def test_rejects_junk_single_flags(self, junk_flags):

        with pytest.raises(TypeError):
            _val_flags(junk_flags, list('ab'))


    @pytest.mark.parametrize('junk_flags',
        (list('ab'), set(list('ab')), tuple(list('ab')))
    )
    def test_rejects_junk_seq_flags(self, junk_flags):

        with pytest.raises(TypeError):
            _val_flags(junk_flags, list('ab'))


    def test_rejects_bad_seq_flags(self):

        # too long
        with pytest.raises(ValueError):
            _val_flags(
                np.random.randint(0,100, (6,)),
                list('abcde')
            )


        # too short
        with pytest.raises(ValueError):
            _val_flags(
                np.random.randint(0,100, (4,)),
                list('abcde')
            )


    def test_accepts_None_int_seq_int(self):


        assert _val_flags(None, list('ab')) is None

        assert _val_flags(
            int(np.random.randint(0,100)),
            list('ab')
        ) is None

        _flags = np.random.randint(0, 100, (5,)).tolist()

        assert _val_flags(list(_flags), list('abcde')) is None

        assert _val_flags(tuple(_flags), list('abcde')) is None

        assert _val_flags(set(_flags), list('abcde')) is None

        assert _val_flags(np.array(_flags), list('abcde')) is None





