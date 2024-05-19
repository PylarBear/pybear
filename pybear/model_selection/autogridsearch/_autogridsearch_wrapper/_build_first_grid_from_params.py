# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#






def _build(_params:dict) -> dict:


    """
    Initialize GRIDS by filling the first round of grids based on the information
    provided in :param: params.

    This is only for the first pass, and no other.


    """

    _GRIDS = {0: {}}

    for _param_ in _params:

        _GRIDS[0][_param_] = _params[_param_][0]


    return _GRIDS




















