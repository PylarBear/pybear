# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from ._type_aliases import ParamsType, GridsType


def _build(_params: ParamsType) -> GridsType:


    """
    Initialize GRIDS by filling the first round of grids based on the information
    provided in :param: params.

    This is only for the first pass, and no other.


    """

    _GRIDS = {0: {}}

    for _param_ in _params:

        _GRIDS[0][_param_] = _params[_param_][0]


    return _GRIDS




















