# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._TextJustifier._validation._join_2D \
    import _val_join_2D

import pytest

import numpy as np



class TestValJoin2D:


    @pytest.mark.parametrize('junk_join_2D',
        (-2.7, -1, 0, 1, 2.7, True, False, None, [0,1], (1, 2), {1,2}, {'a':1},
         lambda x: x)
    )
    def test_junk_join_2D(self, junk_join_2D):
        with pytest.raises(TypeError):
            _val_join_2D(junk_join_2D, _n_rows=2)


    @pytest.mark.parametrize('good_join_2D', (' ', '_', ','))
    def test_good_join_2D_str(self, good_join_2D):

        assert _val_join_2D(good_join_2D, _n_rows=3) is None


    @pytest.mark.parametrize('container', (list, tuple, set, np.array))
    def test_good_join_2D(self, container):

        _base_join2D = list('abc')

        if container is np.array:
            _join2D = np.array(_base_join2D)
        else:
            _join2D = container(_base_join2D)


        assert _val_join_2D(_join2D, _n_rows=3) is None


    @pytest.mark.parametrize('_n_rows', (3, 5, 7))
    def test_rejects_bad_sequence_len(self, _n_rows):

        with pytest.raises(ValueError):
            _val_join_2D(list('abcd'), _n_rows=_n_rows)


        with pytest.raises(ValueError):
            _val_join_2D(list('abcdef'), _n_rows=_n_rows)





