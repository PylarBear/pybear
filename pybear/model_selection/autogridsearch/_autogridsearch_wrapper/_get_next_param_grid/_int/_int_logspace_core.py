# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#





import numpy as np
from typing_extensions import Union
from typing_extensions import TypeAlias
from .._int._int_logspace_gap_gt_1_soft import _int_logspace_gap_gt_1_soft
from .._int._int_logspace_gap_gt_1_hard import _int_logspace_gap_gt_1_hard
from ._int_grid_mapper import _int_grid_mapper


# 24_05_18_18_39_00 _int_logspace_gap_gt_1 was written and tested in
# entirety before this. It turns out that _int_logspace_gap_gt_1 is also
# capable of handling unit log gaps. The original prediction was that
# unit and >1 log gaps would need separate modules, and tests and
# _validation modules were built in anticipation of that. But now that
# both could be passed to the same module, consolidating both operations
# for unit and >1 gaps to that module would require appreciable overhaul
# of existing tests and _validation modules to make the tests and
# _validation handle the passing of both to the same place. So, creating
# a 'core' module with functional code extracted from _int_logspace_gap_gt_1
# that feeds both unit and >1 log gap modules, with individual _validation
# for each of the modules.


# see _type_aliases, subtyes of DataType and GridType
IntDataType: TypeAlias = int
IntGridType: TypeAlias = \
    Union[list[IntDataType], tuple[IntDataType], set[IntDataType]]



def _int_logspace_core(
                        _SINGLE_GRID: IntGridType,
                        _is_logspace: Union[bool, float],
                        _posn: int,
                        _is_hard: bool,
                        _hard_min: IntDataType,
                        _hard_max: IntDataType,
                        _points: int
    ) -> list[IntDataType]:


    """

    Logspaces convert to linspace.

    Build a new grid in linspace for a single integer parameter based on
    the previous search round's logspace grid and the best value
    discovered by GridSearch, subject to constraints imposed by 'hard',
    universal lower bound on integers, etc.

    Parameters
    ----------
    _SINGLE_GRID:
        Union[list[int], tuple[int], set[int]] - The last round's logspace
        search grid for a single parameter. _SINGLE_GRID must be sorted
        ascending, and is presumed to be by _validation._numerical_params
        (at least initially).
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
            list[int] - new linspace search grid for the current pass'
            upcoming search.

    """


    _LOG_SINGLE_GRID = np.log10(_SINGLE_GRID)
    _log_hard_min = np.log10(_hard_min)
    _log_hard_max = np.log10(_hard_max)



    # _left & _right are in logspace!
    if not _is_hard:

        _left, _right = _int_logspace_gap_gt_1_soft(
            _LOG_SINGLE_GRID,
            _is_logspace,
            _posn
        )

    elif _is_hard:
        _left, _right = _int_logspace_gap_gt_1_hard(
            _LOG_SINGLE_GRID,
            _is_logspace,
            _posn,
            _log_hard_min,
            _log_hard_max
        )


    if int(_left) != _left:
        raise ValueError(f"'_left' is not an integer ({_left})")

    if int(_right) != _right:
        raise ValueError(f"'_right' is not an integer ({_right})")

    _left = 10 ** int(_left)
    _right = 10 ** int(_right)

    if _left > _right:
        raise ValueError(f"_left ({_left}) > _right ({_right})")

    if (_right - _left) < 2:
        raise ValueError(f"_right ({_right}) and _left ({_left}) yield "
                         f"less than three points")


    _OUT_GRID = _int_grid_mapper(_left, _right, _points)


    del _left, _right

    _OUT_GRID = list(map(int, _OUT_GRID))

    return _OUT_GRID










