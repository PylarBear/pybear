# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from model_selection.autogridsearch._autogridsearch_wrapper._type_aliases \
    import GridsType, BestParamsType




def _validate_best_params(
        _GRIDS: GridsType,
        _pass: int,
        _best_params_from_previous_pass: BestParamsType,
    ) -> None:


    """
    --- _best_params_from_previous_pass is (still) dict (this is a
        sklearn/dask output and is beyond the control of pybear.)
    --- params (keys) returned in _best_params_from_previous_pass match
        those passed by GRIDS in quantity and values
    --- values returned in best_params were in the allowed search space

    Parameters
    ----------
    _GRIDS:
        dict[int, dict[str, list[...]]] - param_grid for each round
    _pass:
        int - zero-indexed pass of GridSearchCV
    _best_params_from_previous_pass:
        dict[str, [int, float, str]] - best_params_ as returned from
        sklearn / dask GridSearchCV

    Return
    ------
    None


    """


    # best_params_ from dask/sklearn.GridSearchCV looks like
    # {'C': 1, 'l1_ratio': 0.9}

    if not isinstance(_best_params_from_previous_pass, dict):
        raise TypeError(f'best_params_from_previous_pass is not a dict. Has '
                        f'GridSearchCV best_params_ output changed?')


    _, __ = len(_best_params_from_previous_pass), len(_GRIDS[_pass - 1])
    if _ != __:
        raise ValueError(f'len(best_params_from_previous_pass) ({_}) != len(params)'
                         f' from previous pass ({__})')
    del _, __

    for param_ in _best_params_from_previous_pass:
        # VALIDATE best_param_ KEYS WERE IN ITS GRID
        if param_ not in _GRIDS[_pass - 1]:
            raise ValueError(f'{param_} in best_params_from_previous_pass is not '
                             f'in params given by GRIDS on the previous pass')

        # VALIDATE THAT RETURNED best_params_ HAS VALUES THAT ARE WITHIN
        # THE PREVIOUS SEARCH SPACE
        _value = _best_params_from_previous_pass[param_]
        if _value not in _GRIDS[_pass - 1][param_]:
            raise ValueError(f"{param_}: best_params_ contains a value ({_value}) "
                f"that was not in its given search space "
                f"({_GRIDS[_pass - 1][param_]})")


    for param_ in _GRIDS[_pass - 1]:
        # VALIDATE GRID KEYS ARE IN best_params_from_previous_pass
        if param_ not in _best_params_from_previous_pass:
            raise ValueError(f'{param_} in GRIDS[{_pass}] is not in '
                             f'best_params_from_previous_pass')















