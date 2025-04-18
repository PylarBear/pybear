# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.model_selection.autogridsearch._autogridsearch_wrapper._validation. \
    _max_shifts import _val_max_shifts



class TestMaxShifts:


    @pytest.mark.parametrize('non_numeric',
    (True, False, 'string', [1,2], (1,2), {1,2}, {'a':1}, lambda x: x)
    )
    def test_rejects_non_numeric(self, non_numeric):
        with pytest.raises(TypeError):
            _val_max_shifts(non_numeric)


    @pytest.mark.parametrize('non_integer',
        (float('-inf'), -2.718, 2.718, float('inf'))
    )
    def test_rejects_non_integer(self, non_integer):
        with pytest.raises(TypeError):
            _val_max_shifts(non_integer)


    @pytest.mark.parametrize('less_than_one', (0, -1))
    def test_rejects_less_than_one(self, less_than_one):
        with pytest.raises(ValueError):
            _val_max_shifts(less_than_one)


    @pytest.mark.parametrize('_max_shifts', (None, 3, 10, 1_000))
    def test_accepts_good(self, _max_shifts):

        # int >= 1 or None

        assert _val_max_shifts(3) is None









