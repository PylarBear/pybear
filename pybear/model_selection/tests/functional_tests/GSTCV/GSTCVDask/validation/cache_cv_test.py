# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

from model_selection.GSTCV._GSTCVDask._validation._cache_cv import _validate_cache_cv



class TestValidateCacheCV:

    @pytest.mark.parametrize('junk_cachecv',
        (0, 1, 3.14, None, min, 'junk', [0,1], (0,1), {0,1}, {'a':1}, lambda x: x)
    )
    def test_rejects_all_non_bool(self, junk_cachecv):
        with pytest.raises(TypeError):
            _validate_cache_cv(junk_cachecv)



    @pytest.mark.parametrize('good_cachecv', (True, False))
    def test_accepts_bool(self, good_cachecv):
        assert _validate_cache_cv(good_cachecv) is good_cachecv




