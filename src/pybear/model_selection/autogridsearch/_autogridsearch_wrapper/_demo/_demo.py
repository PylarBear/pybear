# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Literal
from typing_extensions import Union

import numbers

from ._make_true_best import _make_true_best
from ._validate_true_best import _validate_true_best
from ._display_true_best import  _display_true_best
from ._mock_gscv import _mock_gscv
from .._build_first_grid_from_params import _build
from .._type_aliases import ParamType, ParamsType, BestParamsType, ResultsType
from .._print_results import _print_results



_params: ParamsType
_IS_LOGSPACE: dict[str, Union[Literal[False], numbers.Real]]
_RESULTS: ResultsType
_pass: int
_param_grid: ParamType
_RESULTS: ResultsType


def _demo(
    _DemoCls,
    _true_best: [None, BestParamsType] = None,
    _mock_gscv_pause_time: numbers.Real = 5
):

    """
    Simulated trials of this AutoGridSearch instance.

    Demonstrate and assess AutoGridSearch's ability to generate
    appropriate grids with the given parameters in this
    AutoGridSearch instance (params, etc.) against mocked true
    best values. Visually inspect the generated grids and
    performance of the AutoGridSearch instance in converging to
    the mock targets provided in true_best_params. If no true
    best values are provided to true_best_params, random true
    best values are generated from the set of first search grids
    provided in params.


    Parameters
    ----------
    _DemoCls:
        Instance of AutoGridSearch created for demo purposes,
        not "self".
    _true_best:
        dict[str, Union[numbers.Real, str]] - dict of mocked true best
        values for an estimator's hyperparameters, as provided by the
        user or generated randomly. If not passed, random true best
        values are generated based on the first round grids made from
        the instructions in params.
    _mock_gscv_pause_time:
        int, float - time in seconds to pause, simulating a trial
        of GridSearch


    Return
    ------
    -
        _DemoCls:
            AutoGridSearchCV instance - The AutoGridSearch instance
            created to run simulations, not the active instance of
            AutoGridSearch. This return is integral for tests of
            the demo functionality, but has no other internal use.

    """


    try:
        float(_mock_gscv_pause_time)
        if _mock_gscv_pause_time < 0:
            raise Exception
    except:
        raise ValueError(f"'_mock_gscv_pause_time' must be a non-negative number")


    # STUFF FOR MIMICKING GridSearchCV.best_params_ ** * ** * ** * ** *
    if _true_best is None:
        _true_best = _make_true_best(_DemoCls.params)

    _validate_true_best(_DemoCls.params, _DemoCls._IS_LOGSPACE, _true_best)

    _true_best_header = f'\nTrue best params'
    print(_true_best_header)
    print(f'-' * len(_true_best_header))
    _display_true_best(_DemoCls.params, _true_best)
    # END STUFF FOR MIMICKING GridSearchCV.best_params_ ** * ** * ** *


    # MIMIC GridSearchCV.fit() FLOW AND OUTPUT
    # fit():
    #             1) run passes of GridSearchCV
    #               - 1a) get_next_param_grid()
    #               - 1b) fit GridSearchCV with next_param_grid
    #               - 1c) update self.RESULTS
    #             2) return best_estimator_

    # 1) run passes of GridSearchCV
    _RESULTS = dict()
    _pass = 0
    while _pass < _DemoCls.total_passes:

        print(f"\nStart pass {_pass + 1} " + f"** * " * 15)

        # short-circuiting around fit() in _DemoCls because estimator
        # must be circumvented. Other functionality in fit() (like build
        # param_grids and update RESULTS) must be replicated separately.

        # 1a) get_next_param_grid()
        print(f'Building param grid... ', end='')
        if _pass == 0:
            _DemoCls.GRIDS_ = _build(_DemoCls.params)
            # points must match what is in params
        else:
            # _DemoCls.total_passes would be updated by gnpg
            _DemoCls._get_next_param_grid(
                _pass,
                _RESULTS[_pass-1]
            )
            # update points in params with possibly different points from gnpg
            for _param in _DemoCls.GRIDS_[_pass]:
                if _DemoCls.params[_param][-1] not in ['string', 'bool']:
                    _DemoCls.params[_param][1][_pass] = \
                                    len(_DemoCls.GRIDS_[_pass][_param])

        print(f'Done.')


        # 1b) fit GridSearchCV with next_param_grid
        _RESULTS[_pass] = _mock_gscv(
            _DemoCls.GRIDS_,
            _DemoCls.params,
            _true_best,
            None if _pass == 0 else _RESULTS[_pass - 1],
            _pass,
            _pause_time=_mock_gscv_pause_time
        )

        print(f"End pass {_pass + 1}   " + f"** * " * 15)

        _pass += 1


    # 1c) update self.RESULTS - this must be set to the DemoCls attribute
    # so that DemoCls knows how to build the next param grid
    _DemoCls.RESULTS_ = _RESULTS

    del _RESULTS, _pass


    print(f'\nRESULTS:')
    print(f'--------')
    _print_results(_DemoCls.GRIDS_, _DemoCls.RESULTS_)


    # DISPLAY THE GENERATED true_best_params AGAIN #####################
    print(_true_best_header)
    print(f'-' * len(_true_best_header))
    _display_true_best(_DemoCls.params, _true_best)
    del _true_best_header
    # END DISPLAY THE GENERATED true_best_params AGAIN #################

    # 2) return best_estimator_ --- DONT HAVE AN ESTIMATOR TO RETURN

    print(f"demo fit successfully completed {_DemoCls.total_passes} pass(es) "
          f"with {_DemoCls._shift_ctr} shift pass(es).")


    return _DemoCls   # for tests purposes only







