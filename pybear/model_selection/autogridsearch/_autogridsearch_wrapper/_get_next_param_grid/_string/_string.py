# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

def _string(
            _param_key:str,
            _param_instructions:list,
            _GRIDS:dict,
            _pass:int,
            _best_params_from_previous_pass:dict
    ) -> dict:

    """
    Update GRIDS for a string parameter based on results from _best_params.

    Parameters
    ----------
    _param_key:
        str,
    _param_instructions:
        list - len _param_instructions must be 3, posn 0 is list,
        posn 2 is 'string'
    _GRIDS:
        dict - _GRIDS must be dict
    _pass:
        int - _pass must be positive int
    _best_params_from_previous_pass:
        dict - must be dict (sklearn best_params_)

    """

    if _pass not in _GRIDS:
        raise ValueError(f"attempting to update a pass that is not in GRIDS")

    if _param_key not in _GRIDS[min(_GRIDS.keys())]:
        raise ValueError(f"attempting to insert a param key that is not in GRIDS")


    if _pass >= _param_instructions[1]:
        # _best_params_from_previous_pass[_param_key] is a single value,
        # wrap with []
        _GRIDS[_pass][_param_key] = [_best_params_from_previous_pass[_param_key]]
    else:
        _GRIDS[_pass][_param_key] = _param_instructions[0]


    return _GRIDS
