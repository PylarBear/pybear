# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.feature_extraction.text._TextJustifierRegExp._validation._backfill_sep \
    import _val_backfill_sep




class TestBackfillSep:


    @pytest.mark.parametrize('junk_backfill_sep',
        (-2.7, -1, 0, 1, 2.7, True, False, None, list('ab'), (1,), {1,2}, {'a':1},
         lambda x: x)
    )
    def test_rejects_junk(self, junk_backfill_sep):

        with pytest.raises(TypeError):
            _val_backfill_sep(junk_backfill_sep)



    @pytest.mark.parametrize('_backfill_sep', (' ', ',', '.', ''))
    def test_rejects_junk(self, _backfill_sep):

        assert _val_backfill_sep(_backfill_sep) is None






