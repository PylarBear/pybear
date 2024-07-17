# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np

from model_selection.GSTCV._GSTCVMixin._validation._thresholds__param_grid \
    import _validate_thresholds__param_grid



class TestValidateThresholdsParamGrid:

    # def _validate_thresholds__param_grid(
    #   _thresholds: Union[Iterable[Union[int, float]], Union[int, float], None],
    #   _param_grid: ParamGridType
    # ) -> ParamGridType:


    @staticmethod
    @pytest.fixture
    def good_threshes():
        return np.linspace(0, 1, 21)


    @staticmethod
    @pytest.fixture
    def good_param_grid():
        return [
            {'thresholds': np.linspace(0,1,11), 'solver':['saga', 'lbfgs']},
            {'solver': ['saga', 'lbfgs'], 'C': np.logspace(-5,5,11)},
            {'thresholds': [0.25], 'solver': ['sage', 'lbfgs'], 'C': [100, 1000]}
        ]



    # param_grid ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('junk_param_grid',
        (0, 1, 3.14, True, False, 'trash', min, lambda x: x)
    )
    def test_rejects_junk_param_grid(self, good_threshes, junk_param_grid):
        with pytest.raises(TypeError):
            _validate_thresholds__param_grid(good_threshes, junk_param_grid)


    @pytest.mark.parametrize('junk_param_grid',
        ({0:1, 1:2}, {'a': 1, 'b': 2}, {0: False, 1: True}, {0:[1,2,3]})
    )
    def test_rejects_junk_dicts(self, good_threshes, junk_param_grid):
        with pytest.raises(TypeError):
            _validate_thresholds__param_grid(good_threshes, junk_param_grid)


    @pytest.mark.parametrize('junk_param_grid',
        ([{0:1, 1:2}], (('a', 1), ('b', 2)), [1,2,3], ['a', 'b', 'c'])
    )
    def test_rejects_junk_lists(self, good_threshes, junk_param_grid):
        with pytest.raises(TypeError):
            _validate_thresholds__param_grid(good_threshes, junk_param_grid)


    def test_accepts_good_param_grids(self, good_threshes, good_param_grid):

        _validate_thresholds__param_grid(good_threshes, None)

        _validate_thresholds__param_grid(good_threshes, good_param_grid)

        _validate_thresholds__param_grid(good_threshes, [good_param_grid[0]])

        _validate_thresholds__param_grid(good_threshes,
            [good_param_grid[0], good_param_grid[1]]
        )

    # END param_grid ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *



    # _thresholds ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @pytest.mark.parametrize('junk_thresh',
        (-1, 3.14, True, False, 'trash', min, [-1,2], (-2,1), lambda x: x)
    )
    def test_rejects_junk_thresh(self, good_param_grid, junk_thresh):
        with pytest.raises((TypeError, ValueError)):
            _validate_thresholds__param_grid(junk_thresh, good_param_grid)


    @pytest.mark.parametrize('_good_thresh',
        (0, 0.5, 1, [0, 0.1, 0.2], (0.8, 0.9, 1.0), None)
    )
    def test_accepts_good_thresh(self, good_param_grid, _good_thresh):
        _validate_thresholds__param_grid(_good_thresh, good_param_grid)

    # END _thresholds ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # if param_grid had valid thresholds in it, it comes out the same as
    # it went in, regardless of passed threshes (dicts 1 & 3)

    # if param_grid was not passed, but thresholds was, should be a param
    # grid with only the thresholds in it

    # if both param_grid and thresholds were not passed, should be one
    # param grid with default thresholds

    # if param_grid was passed and did not have thresholds, should be the
    # same except have given thresholds in it. If thresholds was not
    # passed, default thresholds should be in it. (dict 2)

    # * * * *



    def test_accuracy_1(self, good_threshes, good_param_grid):

        # if param_grid had valid thresholds in it, it comes out the same as
        # it went in, regardless of passed threshes (dicts 1 & 3)

        out = _validate_thresholds__param_grid(
            np.linspace(0,1,5),
            good_param_grid[0]
        )

        assert isinstance(out, list)
        assert len(out) == 1
        assert isinstance(out[0], dict)
        assert np.array_equiv(out[0].keys(), good_param_grid[0].keys())
        for k, v in out[0].items():
            assert np.array_equiv(out[0][k], good_param_grid[0][k])


        out = _validate_thresholds__param_grid(good_threshes, good_param_grid[2])

        assert isinstance(out, list)
        assert len(out) == 1
        assert isinstance(out[0], dict)
        assert np.array_equiv(out[0].keys(), good_param_grid[2].keys())
        for k, v in out[0].items():
            assert np.array_equiv(out[0][k], good_param_grid[2][k])


    def test_accuracy_2(self):

        # if param_grid was not passed, but thresholds was, should be a param
        # grid with only the thresholds in it

        # notice testing pass as set
        out = _validate_thresholds__param_grid({0, 0.25, 0.5, 0.75, 1}, None)

        assert isinstance(out, list)
        assert len(out) == 1
        assert isinstance(out[0], dict)
        assert np.array_equiv(list(out[0].keys()), ['thresholds'])
        assert np.array_equiv(out[0]['thresholds'], np.linspace(0,1,5))

        # notice testing pass as list
        out = _validate_thresholds__param_grid([0, 0.25, 0.5, 0.75, 1], {})

        assert isinstance(out, list)
        assert len(out) == 1
        assert isinstance(out[0], dict)
        assert np.array_equiv(list(out[0].keys()), ['thresholds'])
        assert np.array_equiv(out[0]['thresholds'], np.linspace(0,1,5))


    def test_accuracy_3(self, good_threshes):

        # if both param_grid and thresholds were not passed, should be one
        # param grid with default thresholds

        out = _validate_thresholds__param_grid(None, None)

        assert isinstance(out, list)
        assert len(out) == 1
        assert isinstance(out[0], dict)
        assert np.array_equiv(list(out[0].keys()), ['thresholds'])
        assert np.array_equiv(out[0]['thresholds'], good_threshes)



    def test_accuracy_4(self, good_param_grid, good_threshes):

        # if param_grid was passed and did not have thresholds, should be the
        # same except have given thresholds in it. If thresholds was not
        # passed, default thresholds should be in it. (dict 2)

        # notice testing pass as set
        out = _validate_thresholds__param_grid(
            {0, 0.25, 0.5, 0.75, 1},
            good_param_grid
        )

        assert isinstance(out, list)
        assert len(out) == 3
        for _idx, _param_grid in enumerate(out):
            assert isinstance(out[_idx], dict)
            if _idx == 1:
                assert np.array_equiv(
                    list(out[_idx].keys()),
                    list(good_param_grid[_idx].keys()) + ['thresholds']
                )
                for k,v in out[_idx].items():
                    if k == 'thresholds':
                        assert np.array_equiv(out[_idx][k], np.linspace(0,1,5))
                    else:
                        assert np.array_equiv(v, good_param_grid[1][k])
            else:
                assert np.array_equiv(
                    list(out[_idx].keys()),
                    list(good_param_grid[_idx].keys())
                )
                for k,v in out[_idx].items():
                    assert np.array_equiv(v, good_param_grid[_idx][k])

        # ** * ** *

        out = _validate_thresholds__param_grid(None, good_param_grid)

        assert isinstance(out, list)
        assert len(out) == 3
        for _idx, _param_grid in enumerate(out):
            assert isinstance(out[_idx], dict)
            if _idx == 1:
                assert np.array_equiv(
                    list(out[_idx].keys()),
                    list(good_param_grid[_idx].keys()) + ['thresholds']
                )
                for k,v in out[_idx].items():
                    if k == 'thresholds':
                        assert np.array_equiv(out[_idx][k], good_threshes)
                    else:
                        assert np.array_equiv(v, good_param_grid[1][k])
            else:
                assert np.array_equiv(
                    list(out[_idx].keys()),
                    list(good_param_grid[_idx].keys())
                )
                for k,v in out[_idx].items():
                    assert np.array_equiv(v, good_param_grid[_idx][k])













































