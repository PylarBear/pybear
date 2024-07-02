# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import numpy as np

from typing import Type, Union
from numpy.typing import NDArray

from pybear.utils import permuter

from model_selection.GSTCV._type_aliases import ParamGridType, ScorerWIPType, CVResultsType


"""
pizza, this is the active cv_results_builder in _GSTCV. there are 2 things 
still up in the air:
1) use the sklearn/dask convention when 1 scorer is used, to say 'score' 
    instead of the scorer name (and actual scorer name is used when 2+ 
    scorers.

2) what does PARAM_GRID_KEY do? --- thinking that is a vector that when 
    held up against the 'params' column in 'cv_results_' it gives the 
    index of the param_grid in PARAM_GRIDS that that permutation is 
    associated with.


"""






def _cv_results_builder(
        param_grid: list[ParamGridType],
        cv: int,
        scorer: ScorerWIPType,
        return_train_score: bool
    ) -> tuple[CVResultsType, NDArray[np.uint8]]:

    # pizza, cv_results_ is only used inside the _GSTCV fit method!



    # RULES FOR COLUMNS
    """
        FROM DASK GridSearchCV DOCS:

        dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns,
        that can be imported into a pandas DataFrame.

        For instance the below given table

        param_kernel param_gamma param_degree split0_test_score  …  rank…..
        ‘poly’       –           2            0.8                …  2
        ‘poly’       –           3            0.7                …  4
        ‘rbf’        0.1         –            0.8                …  3
        ‘rbf’        0.2         –            0.9                …  1

        will be represented by a cv_results_ dict of:

        {
            'param_kernel'       : masked_array(data = ['poly', 'poly', 'rbf', 'rbf'], mask = [False False False False]...)
            'param_gamma'        : masked_array(data = [-- -- 0.1 0.2], mask = [ True  True False False]...),
            'param_degree'       : masked_array(data = [2.0 3.0 -- --], mask = [False False  True  True]...),
            'split0_test_score'  : [0.8, 0.7, 0.8, 0.9],
            'split1_test_score'  : [0.82, 0.5, 0.7, 0.78],
            'mean_test_score'    : [0.81, 0.60, 0.75, 0.82],
            'std_test_score'     : [0.02, 0.01, 0.03, 0.03],
            'rank_test_score'    : [2, 4, 3, 1],
            'split0_train_score' : [0.8, 0.7, 0.8, 0.9],
            'split1_train_score' : [0.82, 0.7, 0.82, 0.5],
            'mean_train_score'   : [0.81, 0.7, 0.81, 0.7],
            'std_train_score'    : [0.03, 0.04, 0.03, 0.03],
            'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
            'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
            'mean_score_time'    : [0.007, 0.06, 0.04, 0.04],
            'std_score_time'     : [0.001, 0.002, 0.003, 0.005],
            'params'             : [{'kernel': 'poly', 'degree': 2}, ...],
            }
            NOTE that the key 'params' is used to store a list of
            parameter settings dict for all the parameter candidates.

            The mean_fit_time, std_fit_time, mean_score_time and
            std_score_time are all in seconds.

            {
            # ALWAYS THE SAME
            'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
            'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
            'mean_score_time'    : [0.007, 0.06, 0.04, 0.04],
            'std_score_time'     : [0.001, 0.002, 0.003, 0.005],
            ** **** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
            # THEN ALWAYS 'param_{param}' (UNIQUE PARAMETERS OF ALL PARAM GRIDS in param_grid)
            'param_kernel'       : masked_array(data = ['poly', 'poly', 'rbf', 'rbf'], mask = [False False False False]...)
            'param_gamma'        : masked_array(data = [-- -- 0.1 0.2], mask = [ True  True False False]...),
            'param_degree'       : masked_array(data = [2.0 3.0 -- --], mask = [False False  True  True]...),
            ** **** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
            # THEN ALWAYS params, WHICH FILLS WITH DICTS FOR EVERY POSSIBLE PERMUTATION FOR THE PARAM GRIDS IN param_grid
            # PASS THESE params DICTS TO set_params FOR THE ESTIMATOR
            'params'             : {'kernel': 'poly', 'degree': 2, ...},
            ** **** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
            #THEN
            for metric in scoring:
                suffix = 'score' if len(scoring)==1 else f'{metric}'
                for split in range(cv):
                    f'split{split}_test_{suffix}'
                    E.G.:
                    f'split0_test_score'  : [0.8, 0.7, 0.8, 0.9],
                    f'split1_test_accuracy'  : [0.82, 0.5, 0.7, 0.78],
                THEN ALWAYS
                f'mean_test_{suffix}'    : [0.81, 0.60, 0.75, 0.82],
                f'std_test_{suffix}'     : [0.02, 0.01, 0.03, 0.03],
                f'rank_test_{suffix}'    : [2, 4, 3, 1],

                if return_train_score is True:
                    ** **** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
                    for split in range(cv):
                        f'split{split}_train_{suffix}'
                        E.G.:
                        f'split0_train_score' : [0.8, 0.7, 0.8, 0.9],
                        f'split1_train_accuracy' : [0.82, 0.7, 0.82, 0.5],
                    THEN ALWAYS
                    f'mean_train_{suffix}'   : [0.81, 0.7, 0.81, 0.7],
                    f'std_train_{suffix}'    : [0.03, 0.04, 0.03, 0.03],
                    ** **** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
        }
    """



    # BUILD VECTOR OF COLUMN NAMES ** ** ** ** ** ** ** ** ** ** **
    # columns_dtypes_appender
    def c_d_a(COLUMNS, DTYPES, NEW_COLUMNS_AS_LIST, NEW_DTYPES_AS_LIST):
        COLUMNS += NEW_COLUMNS_AS_LIST
        DTYPES += NEW_DTYPES_AS_LIST
        return COLUMNS, DTYPES

    COLUMNS: list[str] = []
    DTYPES: list[Type[Union[np.float64, object]]] = []

    # FIXED HEADERS
    COLUMNS, DTYPES = c_d_a(
        COLUMNS,
        DTYPES,
        ['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time'],
        [np.float64 for _ in range(4)]
    )


    # PARAM NAMES
    _unq_params = list({'param_' + _ for __ in param_grid for _ in __})
    COLUMNS, DTYPES = c_d_a(
        COLUMNS,
        DTYPES,
        _unq_params,  # notice set!
        [object for _ in _unq_params]
    )
    del _unq_params
    assert len(COLUMNS) == len(DTYPES), "len(COLUMNS) != len(DTYPES)"


    # PARAM DICTS

    COLUMNS, DTYPES = c_d_a(COLUMNS, DTYPES, ['params'], [object])

    # SCORES
    for metric in scorer:

        if len(scorer) == 1:
            metric = 'score'  # pizza think on this --- is it better to violate the dask/sklearn way and actually use the 1 scorer's name?
            COLUMNS, DTYPES = c_d_a(COLUMNS, DTYPES, [f'best_threshold'], [np.float64])
        else:
            COLUMNS, DTYPES = c_d_a(COLUMNS, DTYPES, [f'best_threshold_{metric}'], [np.float64])

        for split in range(cv):
            COLUMNS, DTYPES = c_d_a(
                COLUMNS,
                DTYPES,
                [f'split{split}_test_{metric}'],
                [np.float64]
            )

        COLUMNS, DTYPES = c_d_a(COLUMNS, DTYPES, [f'mean_test_{metric}'], [np.float64])
        COLUMNS, DTYPES = c_d_a(COLUMNS, DTYPES, [f'std_test_{metric}'], [np.float64])
        COLUMNS, DTYPES = c_d_a(COLUMNS, DTYPES, [f'rank_test_{metric}'], [np.float64])

        if return_train_score:
            for split in range(cv):
                COLUMNS, DTYPES = c_d_a(
                    COLUMNS,
                    DTYPES,
                    [f'split{split}_train_{metric}'],
                    [np.float64]
                )

            COLUMNS, DTYPES = c_d_a(COLUMNS, DTYPES, [f'mean_train_{metric}'], [np.float64])
            COLUMNS, DTYPES = c_d_a(COLUMNS, DTYPES, [f'std_train_{metric}'], [np.float64])

    del c_d_a
    # END BUILD VECTOR OF COLUMN NAMES ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    # GET FULL COUNT OF ROWS FOR ALL PERMUTATIONS ACROSS ALL GRIDS
    total_rows = np.sum([np.prod(i) for i in [[len(list(_)) for _ in __.values()] for __ in param_grid]])

    # BUILD cv_results_
    cv_results_ = {}
    for column_name, _dtype in zip(COLUMNS, DTYPES):
        cv_results_[column_name] = \
            np.ma.masked_array(np.empty(total_rows), mask=True, dtype=_dtype)


    PARAM_GRID_KEY: NDArray[np.uint8] = np.empty(total_rows, dtype=np.uint8)

    del COLUMNS, DTYPES, total_rows

    # POPULATE KNOWN FIELDS IN cv_results_ (only columns associated
    # with params) #################################################

    ctr = 0
    for grid_idx, _grid in enumerate(param_grid):
        PARAMS = list(_grid.keys())
        SUB_GRIDS = list(_grid.values())

        PERMUTATIONS = permuter(_grid.values())

        # a permutation IN PERMUTATIONS LOOKS LIKE (grid_idx_param_0,
        # grid_idx_param_1, grid_idx_param_2,....)
        # BUILD INDIVIDUAL param_grids TO GIVE TO estimator.set_params()
        # FROM THE PARAMS IN "PARAMS" AND VALUES KEYED FROM "SUB_GRIDS"
        for TRIAL in PERMUTATIONS:
            trial_param_grid = dict()
            for grid_idx, sub_grid_idx in enumerate(TRIAL):
                cv_results_[f'param_{PARAMS[grid_idx]}'][ctr] = \
                    SUB_GRIDS[grid_idx][sub_grid_idx]
                trial_param_grid[PARAMS[grid_idx]] = SUB_GRIDS[grid_idx][sub_grid_idx]

            cv_results_['params'][ctr] = trial_param_grid

            PARAM_GRID_KEY[ctr] = grid_idx

            ctr += 1

    del PERMUTATIONS, PARAMS, SUB_GRIDS
    del trial_param_grid, ctr




    # END POPULATE KNOWN FIELDS IN cv_results_ (only columns associated
    # with params) #################################################

    return cv_results_, PARAM_GRID_KEY























