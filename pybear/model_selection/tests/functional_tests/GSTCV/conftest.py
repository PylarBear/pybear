# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np

from model_selection.GSTCV._master_scorer_dict import master_scorer_dict
from sklearn.model_selection import ParameterGrid




@pytest.fixture(scope='module')
def _cv_results_template(request):
    # _n_splits, _n_rows, _scorer_names, _grid, _return_train_score, _fill_param_columns

    """
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

    _n_splits = request.param['_n_splits']
    _n_rows = request.param['_n_rows']
    _scorer_names = request.param['_scorer_names']
    _grids = request.param['_grids']
    _return_train_score = request.param['_return_train_score']
    _fill_param_columns = request.param['_fill_param_columns']

    # build _scorer ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    try:
        iter(_scorer_names)
        if isinstance(_scorer_names, (str, dict)):
            raise Exception
    except:
        raise Exception(
            f"'LIST_OF_SCORERS' must be an iterable of scorer names")

    _scorer = {}
    for _name in _scorer_names:
        if _name not in master_scorer_dict:
            raise ValueError(f"'{_name}' is not an allowed scorer")

        _scorer[_name] = master_scorer_dict[_name]

    # END build _scorer ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    col_template = lambda _dtype: np.ma.masked_array(
        np.empty(_n_rows),
        mask=True,
        dtype=_dtype
    )

    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    a = {
        'mean_fit_time': col_template(np.float64),
        'std_fit_time': col_template(np.float64),
        'mean_score_time': col_template(np.float64),
        'std_score_time': col_template(np.float64)
    }
    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    b = {}
    for _grid in _grids:
        for _param in _grid:
            # this will overwrite any identical, preventing duplicate
            b[f'param_{_param}'] = col_template(str)

        b = b | {'params': col_template(object)}

    if _fill_param_columns:
        row_idx = 0
        for _grid in _grids:
            for _permutation in ParameterGrid(_grid):
                # ParameterGrid lays out permutations in the same order as pybear.permuter

                b['params'][row_idx] = _permutation

                for _param in _grid:
                    if f'param_{_param}' not in b:
                        raise Exception
                    b[f'param_{_param}'][row_idx] = _permutation[_param]

                row_idx += 1


    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    c = {}
    for metric in _scorer:
        suffix = 'score' if len(_scorer) == 1 else f'{metric}'

        if len(_scorer) == 1:
            c[f'best_threshold'] = col_template(np.float64)
        else:
            c[f'best_threshold_{metric}'] = col_template(np.float64)

        for split in range(_n_splits):
            c[f'split{split}_test_{suffix}'] = col_template(np.float64)

        c[f'mean_test_{suffix}'] = col_template(np.float64)
        c[f'std_test_{suffix}'] = col_template(np.float64)
        c[f'rank_test_{suffix}'] = col_template(np.uint32)

        if _return_train_score is True:

            for split in range(_n_splits):
                c[f'split{split}_train_{suffix}'] = col_template(np.float64)

            c[f'mean_train_{suffix}'] = col_template(np.float64)
            c[f'std_train_{suffix}'] = col_template(np.float64)
    # ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    return a | b | c









































