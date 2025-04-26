# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.model_selection.GSTCV._GSTCVMixin._validation._param_grid \
    import _val_param_grid



class TestValParamGrid:


    # def _val_param_grid(
    #     _param_grid: Union[ParamGridType, Sequence[ParamGridType], None]
    # ) -> None:



    @staticmethod
    @pytest.fixture(scope='function')
    def good_param_grid():
        return [
            {'thresholds': np.linspace(0,1,11), 'solver':['saga', 'lbfgs']},
            {'solver': ['saga', 'lbfgs'], 'C': np.logspace(-5,5,11)},
            {'thresholds': [0.25], 'solver': ['saga', 'lbfgs'], 'C': [100, 1000]}
        ]



    # param_grid ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('junk_param_grid',
        (0, 1, 3.14, True, False, 'trash', min, lambda x: x)
    )
    def test_rejects_junk_param_grid(self, junk_param_grid):
        with pytest.raises(TypeError):
            _val_param_grid(junk_param_grid)


    @pytest.mark.parametrize('junk_param_grid',
        ({0:1, 1:2}, {'a': 1, 'b': 2}, {0: False, 1: True}, {0:[1,2,3]})
    )
    def test_rejects_junk_dicts(self, junk_param_grid):
        with pytest.raises(TypeError):
            _val_param_grid(junk_param_grid)


    @pytest.mark.parametrize('junk_param_grid',
        ([{0:1, 1:2}], (('a', 1), ('b', 2)), [1,2,3], ['a', 'b', 'c'])
    )
    def test_rejects_junk_lists(self, junk_param_grid):
        with pytest.raises(TypeError):
            _val_param_grid(junk_param_grid)


    def test_accepts_good_param_grids(self, good_param_grid):

        assert _val_param_grid(None) is None

        assert _val_param_grid(good_param_grid) is None

        assert _val_param_grid([good_param_grid[0]]) is None



    @pytest.mark.parametrize('valid_empties',
        ({}, [], [{}], [{}, {}])
    )
    def test_accepts_valid_empties(self, valid_empties):

        assert _val_param_grid(valid_empties) is None


    @pytest.mark.parametrize('bad_empties',
        (((),), ([{}],), [[]], [(), ()])
    )
    def test_rejects_invalid_empties(self, bad_empties):

        with pytest.raises(TypeError):
            _val_param_grid(bad_empties)


    # END param_grid ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *



    # _thresholds ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @pytest.mark.parametrize('junk_thresh',
        (-1, 3.14, True, False, 'trash', min, [-1,2], (-2,1), lambda x: x)
    )
    @pytest.mark.parametrize('grid_idx', (0, 1, 2))
    def test_rejects_junk_thresh(self, good_param_grid, junk_thresh, grid_idx):

        good_param_grid[grid_idx]['thresholds'] = junk_thresh

        with pytest.raises((TypeError, ValueError)):
            _val_param_grid(good_param_grid)


    @pytest.mark.parametrize('good_thresh', ([0, 0.1, 0.2], (0.8, 0.9, 1.0)))
    @pytest.mark.parametrize('grid_idx', (0, 1, 2))
    def test_accepts_good_thresh(self, good_param_grid, good_thresh, grid_idx):

        good_param_grid[grid_idx]['thresholds'] = good_thresh

        _val_param_grid(good_param_grid)



    @pytest.mark.parametrize('thresholds', (None, 0.75, [0.25, 0.5, 0.75]))
    def test_when_thresholds_passed_via_param_grid(self, thresholds):
        # None, single number, cannot be passed as a value for
        # 'thresholds' inside a param_grid, must be passed as a list-like
        # as usual for param grids.

        if thresholds in [None, 0.75]:
            with pytest.raises(TypeError):
                _val_param_grid({'thresholds': thresholds})

        elif np.array_equiv(thresholds, [0.25, 0.5, 0.75]):

            assert _val_param_grid({'thresholds': thresholds}) is None

    # END _thresholds ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *




