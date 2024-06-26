# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from copy import deepcopy
import numpy as np

# PIZZA AS OF 24_02_11_17_57_00 THIS WORKS... KEEP FOR POSTERITY




def cv_results_builder(param_grid, cv, scoring, return_train_score):
    # RULES FOR COLUMNS
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



    # BUILD VECTOR OF COLUMN NAMES ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
    # columns_dtypes_appender
    def c_d_a(COLUMNS, DTYPES, NEW_COLUMNS_AS_LIST, NEW_DTYPES_AS_LIST):
        COLUMNS += NEW_COLUMNS_AS_LIST
        DTYPES += NEW_DTYPES_AS_LIST
        return COLUMNS, DTYPES

    COLUMNS, DTYPES = [], []

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
        _unq_params,
        [object for _ in _unq_params]
    )
    del _unq_params
    assert len(COLUMNS) == len(DTYPES), "len(COLUMNS) != len(DTYPES)"

    # PARAM DICTS
    #'params'  PIZZA FILL THIS FROM PERMUTATIONS
    COLUMNS, DTYPES = c_d_a(COLUMNS, DTYPES, ['params'], [object])

    # SCORES
    for metric in scoring:




        suffix = 'score' if len(scoring)==1 else f'{metric}'


        for split in range(cv):
            COLUMNS, DTYPES = c_d_a(
                COLUMNS,
                DTYPES,
                [f'split{split}_test_{suffix}'],
                [np.float64]
            )

        COLUMNS, DTYPES = c_d_a(COLUMNS, DTYPES, [f'mean_test_{suffix}'], [np.float64])
        COLUMNS, DTYPES = c_d_a(COLUMNS, DTYPES, [f'std_test_{suffix}'], [np.float64])
        COLUMNS, DTYPES = c_d_a(COLUMNS, DTYPES, [f'rank_test_{suffix}'], [np.float64])

        if return_train_score:
            for split in range(cv):
                COLUMNS, DTYPES = c_d_a(
                    COLUMNS,
                    DTYPES,
                    [f'split{split}_train_{suffix}'],
                    [np.float64]
                )

            COLUMNS, DTYPES = c_d_a(COLUMNS, DTYPES, [f'mean_train_{suffix}'], [np.float64])
            COLUMNS, DTYPES = c_d_a(COLUMNS, DTYPES, [f'std_train_{suffix}'], [np.float64])

    del c_d_a
    # END BUILD VECTOR OF COLUMN NAMES ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **

    # GET FULL COUNT OF ROWS FOR ALL PERMUTATIONS ACROSS ALL GRIDS
    total_rows = np.sum([np.prod(i) for i in [[len(list(_)) for _ in __.values()] for __ in param_grid]])

    # BUILD cv_results_
    cv_results_ = {column_name: np.ma.masked_array(np.empty(total_rows), mask=True, dtype=_dtype) for column_name, _dtype
                   in zip(COLUMNS, DTYPES)}




    del COLUMNS, DTYPES

    # POPULATE KNOWN FIELDS IN cv_results_ (only columns associated with params) ##########################################

    def recursive_fxn(cp_vector_of_lens):
        if len(cp_vector_of_lens) == 1:
            seed_array = np.zeros((cp_vector_of_lens[0], len(vector_of_lens)), dtype=int)
            seed_array[:, -1] = range(cp_vector_of_lens[0])
            return seed_array
        else:
            seed_array = recursive_fxn(cp_vector_of_lens[1:])
            stack = np.empty((0, len(vector_of_lens)), dtype=np.uint32)
            for param_idx in range(cp_vector_of_lens[0]):
                filled_array = seed_array.copy()
                filled_array[:, len(vector_of_lens) - len(cp_vector_of_lens)] = param_idx
                stack = np.vstack((stack, filled_array))

            del filled_array
            return stack

    ctr = 0
    for _grid in param_grid:
        PARAMS = list(_grid.keys())
        SUB_GRIDS = list(_grid.values())
        vector_of_lens = list(map(len, SUB_GRIDS))
        cp_vector_of_lens = deepcopy(vector_of_lens)

        PERMUTATIONS = recursive_fxn(cp_vector_of_lens)





        for TRIAL in PERMUTATIONS:
            trial_param_grid = dict()
            for grid_idx, sub_grid_idx in enumerate(TRIAL):
                cv_results_[f'param_{PARAMS[grid_idx]}'][ctr] = SUB_GRIDS[grid_idx][sub_grid_idx]

                trial_param_grid[PARAMS[grid_idx]] = SUB_GRIDS[grid_idx][sub_grid_idx]

            cv_results_['params'][ctr] = trial_param_grid



            ctr += 1

    del recursive_fxn, PERMUTATIONS, PARAMS, SUB_GRIDS, vector_of_lens
    del cp_vector_of_lens, trial_param_grid, ctr

    # DF = pd.DataFrame(cv_results_)
    # print(DF)
    # DF.to_csv(r'/home/bear/Desktop/bear_dump.ods', index=False)
    # END POPULATE KNOWN FIELDS IN cv_results_ (only columns associated with params) #######################################


    return cv_results_











