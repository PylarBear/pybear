# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from copy import deepcopy
import inspect

import numpy as np

from sklearn.model_selection import KFold

from pybear.model_selection.GSTCV._GSTCVMixin._param_conditioning._cv \
    import _cond_cv



class TestCondCV:


    def test_None_returns_default(self):
        _cv = None
        _og_cv = deepcopy(_cv)
        out = _cond_cv(_cv)
        assert isinstance(out, int)
        assert out == 5
        assert _cv is _og_cv


    @pytest.mark.parametrize(f'good_int', (2,3,4,5))
    def test_good_int(self, good_int):

        _og_good_int = deepcopy(good_int)
        out = _cond_cv(good_int)
        assert isinstance(out, int)
        assert out == good_int
        assert good_int == _og_good_int


    @pytest.mark.parametrize(f'junk_iter',
        ([1,2,3], [[1,2,3], [1,2,3], [2,3,4]], (True, False), list('abcde'))
    )
    def test_rejects_junk_iter(self, junk_iter):
        with pytest.raises((TypeError, ValueError)):
            _cond_cv(junk_iter)


    def test_rejects_empties(self):

        with pytest.raises(ValueError):
            _cond_cv([()])


        with pytest.raises(ValueError):
            _cond_cv((_ for _ in range(0)))


    def test_accepts_good_iter(self):

        _n_splits = 3

        X = np.random.randint(0, 10, (20, 5))
        y = np.random.randint(0, 2, 20)
        good_iter = KFold(n_splits=_n_splits).split(X,y)
        # TypeError: cannot pickle 'generator' object
        # cant test that og good_iter is not mutated by _cond_cv
        good_iter2 = KFold(n_splits=_n_splits).split(X,y)

        out = _cond_cv(good_iter)
        assert isinstance(out, list)

        assert inspect.isgenerator(good_iter2)
        iter_as_list = list(good_iter2)
        assert isinstance(iter_as_list, list)

        for idx in range(_n_splits):
            for X_y_idx in range(2):
                assert np.array_equiv(out[idx][X_y_idx], iter_as_list[idx][X_y_idx])





