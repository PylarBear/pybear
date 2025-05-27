# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.preprocessing._ColumnDeduplicateTransformer._partial_fit.\
    _find_duplicates import  _find_duplicates



class TestJoblib:

    # as of 25_05_21 all the tests were passing but things in the joblib
    # part should have been failing. apparently all the other tests are
    # using few enough columns that the joblib part isnt engaged. this
    # test is explicitly designed to engage the joblib code.

    # need to have >= _n_cols number of columns to engage joblib
    # just do minor checks for accuracy


    def test_it_works(self, _X_factory):

        # as of 25_05_24 _n_cols is 200 so use something bigger
        _shape = (623, 538)

        while True:
            _rand_idxs1 = sorted(np.random.randint(0, _shape[1], 5).tolist())
            if len(set(_rand_idxs1)) != len(_rand_idxs1):
                continue
            _rand_idxs2 = sorted(np.random.randint(0, _shape[1], 5).tolist())
            if len(set(_rand_idxs2)) != len(_rand_idxs2):
                continue
            if len(set(_rand_idxs1).intersection(_rand_idxs2)) == 0:
                break

        # this must be sorted asc within inner list and across first
        # idx of each list, because that's how the actual output is
        _dupl = list(sorted([_rand_idxs1, _rand_idxs2]))
        del _rand_idxs1, _rand_idxs2

        _X = _X_factory(
            _dupl=_dupl,
            _has_nan=False,
            _format='np',
            _dtype='flt',
            _columns=None,
            _constants=None,
            _noise=0,
            _zeros=None,
            _shape=_shape
        )


        out = _find_duplicates(
            _X,
             _rtol=1e-5,
             _atol=1e-8,
            _equal_nan=True,
             _n_jobs=-1
        )

        # returns a list of lists
        for idx, dupls in enumerate(out):
            assert np.array_equal(dupls, _dupl[idx])




