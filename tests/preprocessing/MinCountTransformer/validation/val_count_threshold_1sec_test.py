# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.MinCountTransformer._validation. \
    _val_count_threshold import _val_count_threshold

import pytest



class TestValCountThreshold:

    @pytest.mark.parametrize('_count_threshold',
        (True, None, min, [1], (1,), {1,2}, {'a':1}, lambda x: x, 'junk', 3.14)
    )
    def test_typeerror_junk_count_threshold(self, _count_threshold):
        with pytest.raises(TypeError):
            _val_count_threshold(_count_threshold)


    @pytest.mark.parametrize('_count_threshold',
        (-1, 0, 1)
    )
    def test_valueerror_count_threshold(self, _count_threshold):
        with pytest.raises(ValueError):
            _val_count_threshold(_count_threshold)


    @pytest.mark.parametrize('_count_threshold', (2, 3)
    )
    def test_accepts_good_count_threshold(self, _count_threshold):
        _val_count_threshold(_count_threshold)











