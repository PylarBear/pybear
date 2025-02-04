# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import time
from typing_extensions import TypeAlias, Union

import numpy as np

from .._type_aliases import DataType, GridsType, ParamsType, BestParamsType

ParamType: TypeAlias = list[DataType]


def _mock_gscv(
        _GRIDS: GridsType,
        _params: ParamsType,
        _true_best: BestParamsType,
        _best_params: Union[None, BestParamsType],
        _pass: int,
        *,
        _pause_time: [int, float] = 5
    ) -> BestParamsType:


    """
    Simulate the behavior of GridSearchCV. Take a short pause to identify
    the best parameters in a grid based on the underlying true best value.
    For a string parameter, make it 10% chance that the returned "best"
    is non-best option (simulate a discrete parameter moving around while
    the other parameters hone in on their true best.) For numerical, use
    min lsq to find best value.

    Parameters
    ----------
        _GRIDS:
            dict[int, dict[str, list[...]]] - full set of search grids
            for every parameter in every pass
        _params:
            dict[str, list[...]] - full set of grid-building instructions
        _true_best:
            dict[str, [int, float, str]] - the "true best" value for every
            parameter as entered by the user or generated randomly
        _best_params:
            dict[str, [int, float, str] - best results from the previous
            GridSearch pass. "NA" if on pass 0.
        _pass:
            int - the zero-indexed count of GridSearches performed
        _pause_time:
            int - seconds to pause to simulate work by GridSearchCV

    Return
    ------
    -
        _best_params_: dict[str, [int, float, str]] - The values in each
            search grid closest to the true best value.

    """

    err_msg = f"_pause_time must be a number >= 0"
    try:
        float(_pause_time)
        if not _pause_time >= 0:
            raise ValueError
    except TypeError:
        raise TypeError(err_msg)
    except ValueError:
        raise ValueError(err_msg)
    except Exception as e:
        raise Exception(f"_pause_time validation excepted for uncontrolled reason")


    # display info about parameters ** * ** * ** * ** * ** * ** * ** * **
    def padder(words):
        _pad = 11
        try:
            return str(words)[:_pad].ljust(_pad+3)
        except:
            return 'NA'

    # build data header
    print(
        padder('param'),
        padder('type'),
        padder('true_best'),
        padder('prev_best'),
        padder('new_points'),
        padder('next_grid')
    )

    # fill data below header
    for _ in _GRIDS[_pass]:

        print(
            padder(_),
            padder(_params[_][-1]),
            padder(_true_best[_]),
            padder('NA' if _pass == 0 else _best_params[_]),
            padder(len(_GRIDS[_pass][_])),
            end=' '  # to allow add on for grids below
        )

        _grid = _GRIDS[_pass][_]
        try:
            print(f'{list(map(round, _grid, (3 for _ in _grid)))}')  # dont format this!
        except:
            print(f'{_grid}')  # dont format this!
        del _grid

    del padder
    # END display info about parameters ** * ** * ** * ** * ** * ** * **


    # SIMULATE WORK BY GridSearchCV ON AN ESTIMATOR ** * ** * ** * ** *
    combinations = np.prod(list(map(len, _GRIDS[_pass].values())))
    print(f'\nThere are {combinations:,.0f} combinations to run')
    print(f"Simulating GridSearchCV running on pass {_pass + 1}...")
    time.sleep(_pause_time)  # (combinations)
    del combinations


    # CALCULATE WHAT THE best_params_ SHOULD BE BASED ON THE true_best_params.
    _best_params_ = dict()
    for _param in _GRIDS[_pass]:
        _grid = _GRIDS[_pass][_param]
        if len(_grid) == 1:
            _best_params_[_param] = _grid[0]
        elif _params[_param][-1] in ['string', 'bool']:
            # for a str or bool param, make it 10% chance that the returned
            # "best" is non-best option
            _p_best = 0.9
            _p_not_best = (1 - _p_best) / (len(_grid) - 1)
            _p = [0.9 if i == _true_best[_param] else _p_not_best for i in _grid]

            _best_params_[_param] = \
                type(_grid[0])(np.random.choice(_grid, 1, False, p=_p)[0])
            del _p_best, _p_not_best, _p
        else:
            # use min lsq to find best for numerical
            # dont let best value get out of here as an np float or int!
            _LSQ = np.power(
                            np.array(_grid) - _true_best[_param],
                            2,
                            dtype=np.float64
            )
            _best_idx = np.arange(len(_grid))[_LSQ == np.min(_LSQ)][0]
            _best_params_[_param] = _grid[_best_idx]
            del _LSQ, _best_idx

        del _grid
    # END SIMULATE WORK BY GridSearchCV ON AN ESTIMATOR ** * ** * ** * **


    return _best_params_









