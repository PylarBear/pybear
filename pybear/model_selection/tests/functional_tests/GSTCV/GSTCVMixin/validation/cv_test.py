# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest
import inspect
import numpy as np

from sklearn.model_selection import KFold

from model_selection.GSTCV._GSTCVMixin._validation._cv import _validate_cv



class TestValidateCV:


    @pytest.mark.parametrize('junk_cv',
        (2.718, 3.1416, True, False, 'trash', min, {'a': 1}, lambda x: x)
    )
    def test_rejects_non_None_iter_int(self, junk_cv):
        with pytest.raises(TypeError):
            _validate_cv(junk_cv)


    def test_None_returns_default(self):
        assert _validate_cv(None) == 5


    @pytest.mark.parametrize('bad_cv', (-1, 0, 1))
    def test_value_error_less_than_2(self, bad_cv):
        with pytest.raises(ValueError):
            _validate_cv(bad_cv)


    @pytest.mark.parametrize(f'good_int', (2,3,4,5))
    def test_good_int(self, good_int):
        assert _validate_cv(good_int) == good_int


    @pytest.mark.parametrize(f'junk_iter',
        ([1,2,3], [[1,2,3], [1,2,3], [2,3,4]], (True, False), list('abcde'))
    )
    def test_rejects_junk_iter(self, junk_iter):
        with pytest.raises(TypeError):
            _validate_cv(junk_iter)


    def test_accepts_good_iter(self):

        _n_splits = 3

        X = np.random.randint(0, 10, (20, 5))
        y = np.random.randint(0, 2, 20)
        good_iter = KFold(n_splits=_n_splits).split(X,y)
        good_iter2 = KFold(n_splits=_n_splits).split(X,y)

        out = _validate_cv(good_iter)
        assert isinstance(out, list)
        assert inspect.isgenerator(good_iter2)
        iter_as_list = list(good_iter2)
        assert isinstance(iter_as_list, list)

        for idx in range(_n_splits):
            for X_y_idx in range(2):
                assert np.array_equiv(
                    out[idx][X_y_idx],
                    iter_as_list[idx][X_y_idx]
                )


    def test_rejects_empties(self):

        with pytest.raises(TypeError):
            _validate_cv([()])


        with pytest.raises(ValueError):
            _validate_cv((_ for _ in range(0)))























