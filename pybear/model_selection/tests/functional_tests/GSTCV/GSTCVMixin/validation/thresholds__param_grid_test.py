# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np

from pybear.model_selection.GSTCV._GSTCVMixin._validation._thresholds__param_grid \
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
            {'thresholds': [0.25], 'solver': ['saga', 'lbfgs'], 'C': [100, 1000]}
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

        out = _validate_thresholds__param_grid(
            good_threshes,
            [good_param_grid[0]]
        )
        # verify thresholds passed via param grid supersede thresholds passed
        # via kwarg (which would land in the first position in the function)
        assert np.array_equiv(out[0]['thresholds'], np.linspace(0,1,11))


        out = _validate_thresholds__param_grid(good_threshes,
            [good_param_grid[0], good_param_grid[1]]
        )
        # verify thresholds passed via param grid supersede thresholds passed
        # via kwarg (which would land in the first position in the function)
        assert np.array_equiv(out[0]['thresholds'], np.linspace(0,1,11))
        # grid #2 doesnt have thresholds passed, so should be the default
        assert np.array_equiv(out[1]['thresholds'], np.linspace(0,1,21))



    @pytest.mark.parametrize('valid_empties',
        ({}, [], [{}], [{}, {}])
    )
    def test_accepts_valid_empties(self, valid_empties):

        _thresholds = {0, 0.25, 0.5, 0.75, 1}

        out = _validate_thresholds__param_grid(
            _thresholds,
            valid_empties
        )

        # remember validation always returns a list of dicts
        for _grid in out:
            assert len(_grid) == 1
            assert 'thresholds' in _grid
            __ = _grid['thresholds']
            assert isinstance(__, np.ndarray)
            assert np.array_equiv(__, np.array(sorted(list(_thresholds))))


    @pytest.mark.parametrize('bad_empties',
        (((),), ([{}],), [[]], [(), ()])
    )
    def test_rejects_invalid_empties(self, bad_empties):

        with pytest.raises(TypeError):
            _validate_thresholds__param_grid(None, bad_empties)


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
    # it went in, regardless of what is passed to threshold kwarg (dicts 1 & 3)

    # if param_grid was not passed, but thresholds was, should be a param
    # grid with only the thresholds in it

    # if both param_grid and thresholds were not passed, should be one
    # param grid with default thresholds

    # if param_grid was passed and did not have thresholds, should be the
    # same except have given thresholds in it. If thresholds was not
    # passed, default thresholds should be in it. (dict 2)

    # * * * *


    # conditionals between param_grid and thresholds ** * ** * ** * ** *

    def test_accuracy_1(self, good_threshes, good_param_grid):

        # if param_grid had valid thresholds in it, it comes out the same as
        # it went in, regardless of what is passed to threshold kwarg
        # (dicts 1 & 3)

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
        # thresholds passed via param grid supersede those passed via kwarg
        assert np.array_equiv(out[0]['thresholds'], np.linspace(0,1,11))

        out = _validate_thresholds__param_grid(good_threshes, good_param_grid[2])

        assert isinstance(out, list)
        assert len(out) == 1
        assert isinstance(out[0], dict)
        assert np.array_equiv(out[0].keys(), good_param_grid[2].keys())
        for k, v in out[0].items():
            assert np.array_equiv(out[0][k], good_param_grid[2][k])
        assert np.array_equiv(out[0]['thresholds'], [0.25])


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
                        assert np.array_equiv(v, good_param_grid[_idx][k])
            else:
                assert np.array_equiv(
                    list(out[_idx].keys()),
                    list(good_param_grid[_idx].keys())
                )
                for k,v in out[_idx].items():
                    assert np.array_equiv(v, good_param_grid[_idx][k])


    @pytest.mark.parametrize('thresholds', (None, 0.75, [0.25, 0.5, 0.75]))
    def test_when_thresholds_passed_via_param_grid(self, thresholds):
        # None, single number, cannot be passed as a value for
        # 'thresholds' inside a param_grid, must be passed as a vector-
        # like as usual for param grids.

        if thresholds in [None, 0.75]:
            with pytest.raises(TypeError):
                _validate_thresholds__param_grid(
                    None,
                    {'thresholds': thresholds}
                )


        elif np.array_equiv(thresholds, [0.25, 0.5, 0.75]):

            out = _validate_thresholds__param_grid(
                None,
                {'thresholds': thresholds}
            )

            assert np.array_equiv(
                out[0]['thresholds'],
                [0.25, 0.5, 0.75]
            )


































