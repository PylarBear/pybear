# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from ..._type_aliases import GridsType


def _validate_grids(_GRIDS: GridsType) -> None:

    """
    Validate _GRIDS is dict[int: dict[str, [int, float, bool, str]]]


    Parameters
    ----------
    _GRIDS: dict[int, dict[str, [str, int, bool, float]]]] - search grids for
        completed GridSearchCV passes

    Return
    ------
    None

    """


    err_msg = f"_GRIDS must be a dict with int keys and dict values"
    if not isinstance(_GRIDS, dict):
        raise TypeError(err_msg)

    # all keys must be ints
    if any(map(isinstance, _GRIDS.keys(), (bool for _ in _GRIDS))):
        raise TypeError(err_msg)
    if not all(map(isinstance, _GRIDS.keys(), (int for _ in _GRIDS))):
        raise TypeError(err_msg)

    # all values must be dicts
    if not all(map(isinstance, _GRIDS.values(), (dict for _ in _GRIDS))):
        raise TypeError(err_msg)
    del err_msg


    err_msg = (f"_GRIDS values must be dicts with str keys and list values "
               f"holding ints, floats, bools, or strs")
    # all inner dicts keys must be strs
    for pass_idx in _GRIDS:

        __ = _GRIDS[pass_idx]

        if __ == {}:
            continue

        # all inner dicts keys must be str
        if not all(map(isinstance, __.keys(),  (str for _ in __))):
            raise TypeError(err_msg)

        # all inner dicts values must be list-types
        if not all(map(isinstance, __.values(), (list for _ in __))):
            raise TypeError(err_msg)

        # all grids must contain int, float, bool, or str
        for _param, _grid in __.items():
            try:
                if not all(map(isinstance, _grid, (int for _ in _grid))):
                    raise Exception
            except:
                try:
                    if not all(map(isinstance, _grid, (float for _ in _grid))):
                        raise Exception
                except:
                    try:
                        if not all(map(isinstance, _grid, (str for _ in _grid))):
                            raise Exception
                    except:
                        if not all(map(isinstance, _grid, (bool for _ in _grid))):
                            _more_info = (f"\n -- pass {pass_idx}, param '{_param}', "
                                f"grid = {_grid}, types = {list(map(type, _grid))}")
                            raise TypeError(err_msg + _more_info)










