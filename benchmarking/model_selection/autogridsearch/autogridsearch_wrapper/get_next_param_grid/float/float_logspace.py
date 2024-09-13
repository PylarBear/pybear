# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from model_selection.autogridsearch._autogridsearch_wrapper._get_next_param_grid. \
    _float._float_logspace import _float_logspace

import numpy as np


_GRID = list(map(float, np.logspace(-5,5,6).tolist()))
_POSN = {'left':0, 'right':len(_GRID) - 1, 'middle':1}
_hard_min = _GRID[0]
_hard_max = _GRID[-1]

_is_hard = False
_points = 19
_is_logspace = 2


for _posn_ in ('left', 'right', 'middle'):

    _posn = _POSN[_posn_]

    out_grid = _float_logspace(
                                _GRID,
                                _posn,
                                _is_logspace,
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















