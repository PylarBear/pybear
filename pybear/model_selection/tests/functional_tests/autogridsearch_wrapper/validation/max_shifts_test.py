# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

from model_selection.autogridsearch._autogridsearch_wrapper._validation. \
    _max_shifts import _max_shifts


class TestMaxShifts:

    @pytest.mark.parametrize('non_numeric',
    (True, False, None, 'string', [1,2], (1,2), {1,2}, lambda x: x, {'a':1})
    )
    def test_rejects_non_numeric(self, non_numeric):
        with pytest.raises(TypeError):
            _max_shifts(non_numeric)


    def test_rejects_non_integer(self):
        with pytest.raises(TypeError):
            _max_shifts(3.1415)


    @pytest.mark.parametrize('less_than_one',
        (0, -1)
    )
    def test_rejects_less_than_one(self, less_than_one):
        with pytest.raises(ValueError):
            _max_shifts(less_than_one)


    def test_rejects_above_100(self):
        with pytest.raises(ValueError):
            _max_shifts(200)


    def test_accepts_good_positive_integer(self):
        assert _max_shifts(3) == 3
        assert _max_shifts(10) == 10




