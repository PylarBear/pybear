# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np

from model_selection.GSTCV._GSTCV._validation._pre_dispatch import \
    _validate_pre_dispatch


class TestValidatePreDispatch:

    @pytest.mark.parametrize('junk_pd',
        (True, False, min, [0,1], (0,1), {0,1}, {'a':1}, lambda x: x)
    )
    def test_rejects_all_not_none_int_or_str(self, junk_pd):
        with pytest.raises(TypeError):
            _validate_pre_dispatch(junk_pd)


    @pytest.mark.parametrize('junk_pd',
        (np.pi, -np.pi, 0, -1)
    )
    def test_rejects_non_positive_integers(self, junk_pd):
        with pytest.raises(ValueError):
            _validate_pre_dispatch(junk_pd)


    def test_accepts_none(self):
        assert _validate_pre_dispatch(None) is None


    @pytest.mark.parametrize('good_pd', (1, 2, 3, 32))
    def test_accepts_positive_integer(self, good_pd):
        assert _validate_pre_dispatch(good_pd) == good_pd



    @pytest.mark.parametrize('bad_strings',
        ('2*garbage', 'trash+1', 'junk', 'rubbish')
    )
    def test_rejects_bad_strings(self, bad_strings):
        with pytest.raises(ValueError):
            _validate_pre_dispatch(bad_strings)


    @pytest.mark.parametrize('good_strings',
        ('2*n_jobs', 'n_jobs+1', '2*n_jobs-1')
    )
    def test_accepts_functions_of_njobs(self, good_strings):
        n_jobs = 4
        assert _validate_pre_dispatch(good_strings) == good_strings
























