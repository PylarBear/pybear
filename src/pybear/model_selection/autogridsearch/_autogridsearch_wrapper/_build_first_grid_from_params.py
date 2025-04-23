# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from ._type_aliases import ParamsType, GridsType



def _build(_params: ParamsType) -> GridsType:


    """
    Initialize GRIDS by filling the first round of grids based on the
    information provided in :param: `params`.

    This is only for the first pass, and no other. After that, GRIDS
    are built by _get_next_param_grid.


    """


    return {0: {k: v[0] for k, v in _params.items()}}







