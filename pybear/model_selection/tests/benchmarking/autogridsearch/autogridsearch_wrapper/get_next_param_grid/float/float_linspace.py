# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from model_selection.autogridsearch._autogridsearch_wrapper._get_next_param_grid. \
    _float._float_linspace import _float_linspace

import numpy as np


_GRID = list(map(float, np.linspace(0,100,11).tolist()))
_POSN = {'left':0, 'right':len(_GRID) - 1, 'middle':1}
_hard_min = _GRID[0]
_hard_max = _GRID[-1]

_is_hard = False
_points = 19

for _posn_ in ('left', 'right', 'middle'):

    _posn = _POSN[_posn_]

    out_grid = _float_linspace(
                                _GRID,
                                _posn,
                                _is_hard,
                                _hard_min,
                                _hard_max,
                                _points
    )

    print(f"** * " * 10)

    print(f"posn = {_posn_}: ")
    print()
    print(out_grid)
    print(f"** * " * 10)















