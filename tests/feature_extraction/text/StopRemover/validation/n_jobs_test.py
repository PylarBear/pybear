# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.feature_extraction.text._StopRemover._validation._n_jobs import \
    _val_n_jobs



class TestNJobs:


    @pytest.mark.parametrize('junk_n_jobs',
        (-2.7, 2.7, True, False, 'junk', [0,1], (1,), {1,2}, {'A':1}, lambda x: x)
    )
    def test_rejects_junk_n_jobs(self, junk_n_jobs):

        with pytest.raises(TypeError):
            _val_n_jobs(junk_n_jobs)


    @pytest.mark.parametrize('bad_n_jobs', (-2, 0))
    def test_rejects_bad_n_jobs(self, bad_n_jobs):

        with pytest.raises(ValueError):
            _val_n_jobs(bad_n_jobs)


    @pytest.mark.parametrize('_n_jobs', (-1, 1, 4, None))
    def test_accepts_good(self, _n_jobs):

        assert _val_n_jobs(_n_jobs) is None







