# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._TextJustifier._validation._join_2D \
    import _val_join_2D

import pytest



class TestValJoin2D:


    @pytest.mark.parametrize('junk_join_2D',
        (-2.7, -1, 0, 1, 2.7, True, False, None, [0,1], (1, 2), {1,2}, {'a':1},
         lambda x: x)
    )
    def test_junk_join_2D(self, junk_join_2D):
        with pytest.raises(TypeError):
            _val_join_2D(junk_join_2D)


    @pytest.mark.parametrize('good_join_2D', (' ', '_', ','))
    def test_good_join_2D_str(self, good_join_2D):

        assert _val_join_2D(good_join_2D) is None






