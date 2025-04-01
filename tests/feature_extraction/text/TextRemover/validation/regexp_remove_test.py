# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._TextRemover._validation._regexp_remove \
    import _val_regexp_remove

import re

import pytest



class TestValRegExpRemove:


    # def _val_regexp_remove(
    #     _rr: Union[None, re.Pattern[str], tuple[re.Pattern[str], ...],
    #             list[Union[None, re.Pattern[str], tuple[re.Pattern[str], ...]]]],
    #     _n_rows: numbers.Integral
    # ) -> None:


    @pytest.mark.parametrize('junk_bad_n_rows',
        (-2.7, -1, 2.7, True, False, None, 'trash', [0,1], (1,),
         {1,2}, {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_bad_n_rows(self, junk_bad_n_rows):

        with pytest.raises(AssertionError):
            _val_regexp_remove(None, junk_bad_n_rows)


    @pytest.mark.parametrize('good_n_rows', (0, 2, 3, 100_000_000))
    def test_accepts_good_n_rows(self, good_n_rows):

        assert _val_regexp_remove(None, good_n_rows) is None


    @pytest.mark.parametrize('junk_rr',
        (-2.7, -1, 0, 1, 2.7, True, False, 'trash', [0,1], (1,),
         {1,2}, {'a':1}, lambda x: x, tuple('abc'), (list('abc')))
    )
    def test_rejects_junk_rr(self, junk_rr):

        with pytest.raises(TypeError):
            _val_regexp_remove(junk_rr, 5)


    def test_rejects_bad_rr(self):

        _bad_rr = [re.compile('a'), re.compile('b'), re.compile('c')]

        # too long
        with pytest.raises(ValueError):
            _val_regexp_remove(_bad_rr, 2)

        # too short
        with pytest.raises(ValueError):
            _val_regexp_remove(_bad_rr, 4)


    @pytest.mark.parametrize('good_rr',
        (None, re.compile('a'), (re.compile('b'), re.compile('c', re.I)),
        [None, re.compile('a'), (re.compile('b'), re.compile('c', re.I))])
    )
    def test_accepts_good_rr(self, good_rr):

        assert _val_regexp_remove(good_rr, 3) is None



