# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

from model_selection.GSTCV._GSTCVMixin._validation._n_jobs import _val_n_jobs



class TestNJobs:

    @pytest.mark.parametrize('junk_njobs',
        (True, False, 'trash', min, [0,1], (0,1), {0,1}, {'a':1}, lambda x: x)
    )
    def test_rejects_non_int_non_None(self, junk_njobs):

        with pytest.raises(ValueError):
            _val_n_jobs(junk_njobs)

    @pytest.mark.parametrize('bad_njobs',
        (-2, 0, 3.14, float('inf'))
    )
    def test_rejects_bad_int(self, bad_njobs):

        with pytest.raises(ValueError):
            _val_n_jobs(bad_njobs)


    def test_None_returns_one(self):
        assert _val_n_jobs(None) is None


    @pytest.mark.parametrize('good_njobs',
        (-1, 1, 5, 10)
    )
    def test_otherwise_returns_given(self, good_njobs):
        assert _val_n_jobs(good_njobs) == good_njobs




