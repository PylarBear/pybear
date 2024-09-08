# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import sys
from typing import Union, TypeAlias
from utilities._get_module_name import get_module_name
from model_selection.autogridsearch._autogridsearch_wrapper._get_next_param_grid. \
    _validation._validate_int_float_linlogspace import _validate_int_float_linlogspace
from .._int._int_linspace_gap_gt_1_soft import _int_linspace_gap_gt_1_soft
from .._int._int_linspace_gap_gt_1_hard import _int_linspace_gap_gt_1_hard
from ._int_grid_mapper import _int_grid_mapper


# see _type_aliases; subtypes fo DataType, GridType
IntDataType: TypeAlias = int
IntGridType: TypeAlias = \
    Union[list[IntDataType], tuple[IntDataType], set[IntDataType]]



def _int_linspace_gap_gt_1(
                            _SINGLE_GRID: IntGridType,
                            _posn: int,
                            _is_hard: bool,
                            _hard_min: IntDataType,
                            _hard_max: IntDataType,
                            _points: int
    ) -> list[IntDataType]:


    """
    Build a new grid for a single integer parameter with non-unit gaps
    based on the previous search round's grid and the best value
    discovered by GridSearch, subject to constraints imposed by 'hard',
    etc.

    Parameters
    ----------
    _SINGLE_GRID:
        Union[list[int], tuple[int], set[int]] - The last round's search
        grid for a single parameter. _SINGLE_GRID must be sorted ascending,
        and is presumed to be by _validation._numerical_params (at least
        initially).
    _posn:
        int - the index position in the previous round's grid where
        the best value fell
    _is_hard:
        bool - whether the parameter has hard left and right boundaries.
        This field is read from the dtype/search field in _params. If
        hard, the left and right bounds are set from the lowest and
        highest values in the first round's search grid (the grid that
        is in _params.)
    _hard_min:
        int - The minimum value in the first round's search grid. Ignored
        if not hard.
    _hard_max:
        int - The maximum value in the first round's search grid. Ignored
        if not hard.
    _points:
        int - The target number of points for the next search grid. This
        number may not be achieved exactly on ranges that are not evenly
        divisible.

    Return
    ------
    -
        _OUT_GRID:
            list[int] - new search grid for the current pass' upcoming search.


    """

    # 24_05_17_10_04_00 _validation must stay here to get the module name,
    # cannot put in _int
    _SINGLE_GRID =  _validate_int_float_linlogspace(
            _SINGLE_GRID,
            False,
            _posn,
            _is_hard,
            _hard_min,
            _hard_max,
            _points,
            get_module_name(str(sys.modules[__name__]))
    )

    if _is_hard:
        _left, _right = _int_linspace_gap_gt_1_hard(
            _SINGLE_GRID,
            _posn,
            _hard_min,
            _hard_max
        )
    else:
        _left, _right = _int_linspace_gap_gt_1_soft(
            _SINGLE_GRID,
            _posn
        )





    if int(_left) != _left:
        raise ValueError(f"'_left' is not an integer ({_left})")


    if int(_right) != _right:
        raise ValueError(f"'_right' is not an integer ({_right})")

    _left = int(_left)
    _right = int(_right)

    if _left > _right:
        raise ValueError(f"_left ({_left}) > _right ({_right})")


    if _right - _left == 0:
        raise ValueError(f"_right ({_right}) == _left ({_left})")
    elif _right - _left == 1:
        if _posn == 0:
            _right += 1
        elif _posn == len(_SINGLE_GRID) - 1:
            _left -= 1
        else:
            raise ValueError(f"_right ({_right}) - _left ({_left}) "
                                     f"== 1 and not on an edge")
    else:
        pass

    _OUT_GRID = _int_grid_mapper(
        _left,
        _right,
        _points
    )

    del _left, _right

    _OUT_GRID = list(map(int, _OUT_GRID))

    return _OUT_GRID















