# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.MinCountTransformer._validation. \
    _max_recursions import _val_max_recursions

import pytest



class TestValMaxRecursions:

    @pytest.mark.parametrize('_max_recursions',
        (True, None, min, [1], (1,), {1,2}, {'a':1}, lambda x: x, 'junk', 3.14)
    )
    def test_typeerror_junk_max_recursions(self, _max_recursions):
        with pytest.raises(TypeError):
            _val_max_recursions(_max_recursions)


    @pytest.mark.parametrize('_max_recursions',
        (-1, 0, 200)   # allowed is range [1, 100]
    )
    def test_valueerror_max_recursions(self, _max_recursions):
        with pytest.raises(ValueError):
            _val_max_recursions(_max_recursions)


    @pytest.mark.parametrize('_max_recursions', (1, 2, 3)
    )
    def test_accepts_good_max_recursions(self, _max_recursions):
        _val_max_recursions(_max_recursions)




























