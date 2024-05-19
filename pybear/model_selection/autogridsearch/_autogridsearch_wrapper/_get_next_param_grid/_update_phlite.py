# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import numpy as np



def _update_phlite(
                    _PHLITE:dict,
                    _last_param_grid:dict,  # _GRIDS[_pass-1]
                    _params:dict,
                    _best_params_from_previous_pass:dict
    ) -> dict:

    """
    Update PHLITE (PARAM_HAS_LANDED_INSIDE_THE_EDGES) based on most recent
    results in _best_params_from_previous_pass subject to the rules for
    "landing inside the edges".

    The only params that populate PHLITE are soft linspace & soft logspace.

    ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    Rules for landing inside the edges

    1) if soft landed inside the edges, then truly landed "inside the edges"
        and won't be shifted (True)
    2) if only one option in its grid, cannot be shifted (True)
    3) if landed on an edge, but that edge is a universal hard bound
        (0 FOR float, 1 FOR int) then won't be shifted (True)
    4) if landed on an edge, but that edge is not a universal hard bound,
        user hard bound, or fixed, then shift (stays or re-becomes False)
        (user hard or user fixed cannot get in here because PHLITE is only
        populated with soft.)



    """





    for _param in _PHLITE:

        if 'soft' not in _params[_param][-1]:
            raise ValueError(f"'PHLITE' has a non-soft parameter in it --- "
                             f"{_param}: {_params[_param][-1]}")

        _best = _best_params_from_previous_pass[_param]
        _grid = _last_param_grid[_param]

        _edge_finder = np.abs(np.array(_grid) - _best)
        if min(_edge_finder) == _edge_finder[0] or \
                min(_edge_finder) == _edge_finder[-1]:
            _PHLITE[_param] = True

        else:  # MUST BE ON AN EDGE
            if len(_grid) == 1:
                _PHLITE[_param] = True
            elif _params[_param][-1] == 'soft_integer' and _best == 1:
                _PHLITE[_param] = True
            elif _params[_param][-1] == 'soft_float' and _best == 0:
                _PHLITE[_param] = True
            else:
                _PHLITE[_param] = False

    try:
        del _best, _grid, _edge_finder
    except:
        pass


    return _PHLITE











