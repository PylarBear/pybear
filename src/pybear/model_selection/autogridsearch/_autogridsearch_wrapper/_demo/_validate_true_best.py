# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing_extensions import Union

from .._type_aliases import ParamsType, BestParamsType




def _validate_true_best(
        _params: ParamsType,
        _IS_LOGSPACE: dict[str, Union[bool, float]],
        _true_best: BestParamsType
    ) -> None:

    """
    Ensure _true_best matches sklearn / dask GridSearchCV best_params_
    format and best values are valid for the given starting search grids.
    _make_true_best() may intentionally put some best values outside of
    starting search grid to verify autogridsearch's ability to shift.

    Parameters
    ----------
    _params: dict[str, list[...]] - full set of grid-building instructions
    for all params

    _IS_LOGSPACE: dict[bool, float] - False | float for the full set of
    parameters indicating if is not logspace or if logspace, what the
    logspace gap is.

    _true_best: dict[str, [int, float, str]] -

    """

    for _param in _true_best:
        if _param not in _params:
            raise ValueError(f"true_best key '{_param}' not in _params")


    for _param in _params:

        if _param not in _true_best:
            raise ValueError(f"{_param}: true_best must contain all "
                f"the parameters that are in _params")

        _type = _params[_param][-1].lower()
        _grid = _params[_param][0]
        _best = _true_best[_param]

        if 'string' in _type:
            err_msg = (f"{_param}: true_best string params must "
                        f"be a single string that is in the search grid")
            if not isinstance(_best, str):
                raise TypeError(err_msg)
            if _best not in _grid:
                raise ValueError(err_msg)
            del err_msg

        else:

            try:
                float(_best)
            except:
                raise TypeError(f"{_param}: true_best num params must be "
                                 f"a single number")

            if _IS_LOGSPACE[_param] and _best < 0:
                raise ValueError(
                    f"{_param}: true_best must be > 0 for logspace"
                )

            if 'fixed' in _type and _best not in _grid:
                raise ValueError(
                    f"{_param}: 'fixed' numerical true_best must be a "
                    f"single number that is in the search grid"
                )
            elif 'hard' in _type:
                if _best < min(_grid) or _best > max(_grid):
                    raise ValueError(
                        f"{_param}: 'hard' numerical true_best must be in "
                        f"range of given allowed values"
                    )
            else:  # IS SOFT NUMERIC
                pass


            if 'integer' in _type and _best < 1:
                raise ValueError(
                    f"{_param}: soft integer best value must be >= 1"
                )
            elif 'float' in _type and _best < 0:
                raise ValueError(
                    f"{_param}: soft float best value must be >= 0"
                )


    del _type, _grid, _best








