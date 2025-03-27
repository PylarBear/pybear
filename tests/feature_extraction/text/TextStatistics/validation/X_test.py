# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._TextStatistics._validation._X import \
    _val_X

import pytest



class TestValX:


    # validation of check_1D_str_sequence handled on its own


    def test_rejects_empty(self):

        with pytest.raises(ValueError):
            _val_X([])





