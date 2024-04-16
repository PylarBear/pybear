import numpy as np
import pandas as pd
from copy import deepcopy

# BEAR AS OF 24_02_11_17_57_00 THIS WORKS... KEEP FOR POSTERITY


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
    COLUMNS, DTYPES = c_d_a(COLUMNS, DTYPES, ['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time'],
                            [np.float64 for _ in range(4)]
                            )

    # PARAM NAMES
    COLUMNS, DTYPES = c_d_a(COLUMNS, DTYPES, list({'param_' + _ for __ in param_grid for _ in __}),
                            [object for _ in COLUMNS]
                            )

    # PARAM DICTS
    #'params'  BEAR FILL THIS FROM PERMUTATIONS
    COLUMNS, DTYPES = c_d_a(COLUMNS, DTYPES, ['params'], [object])

    # SCORES
    for metric in scoring:
        suffix = 'score' if len(scoring)==1 else f'{metric}'
        for split in range(cv):
            COLUMNS, DTYPES = c_d_a(COLUMNS, DTYPES, [f'split{split}_test_{suffix}'], [np.float64])

        COLUMNS, DTYPES = c_d_a(COLUMNS, DTYPES, [f'mean_test_{suffix}'], [np.float64])
        COLUMNS, DTYPES = c_d_a(COLUMNS, DTYPES, [f'std_test_{suffix}'], [np.float64])
        COLUMNS, DTYPES = c_d_a(COLUMNS, DTYPES, [f'rank_test_{suffix}'], [np.float64])

        if return_train_score:
            for split in range(cv):
                COLUMNS, DTYPES = c_d_a(COLUMNS, DTYPES, [f'split{split}_train_{suffix}'], [np.float64])

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

    del recursive_fxn, PERMUTATIONS, PARAMS, SUB_GRIDS, vector_of_lens, cp_vector_of_lens, trial_param_grid, ctr

    # DF = pd.DataFrame(cv_results_)
    # print(DF)
    # DF.to_csv(r'/home/bear/Desktop/bear_dump.ods', index=False)
    # END POPULATE KNOWN FIELDS IN cv_results_ (only columns associated with params) #######################################

    return cv_results_









if __name__ == '__main__':

    # TEST FOR cv_results_builder ######################################################################################
    print(f'RUNNING cv_results_builder TEST')

    param_grid = [
                    {'kernel': ['rbf'], 'gamma': [0.1, 0.2], 'test_param': [1, 2, 3]},
                    {'kernel': ['poly'], 'degree': [2, 3], 'test_param': [1, 2, 3]},
    ]

    correct_cv_results_len = np.sum(list(map(np.prod, [[len(_) for _ in __] for __ in map(dict.values, param_grid)])))


    CV = [3,4,5]

    SCORING = [['accuracy'], ['accuracy', 'balanced_accuracy']]

    RETURN_TRAIN = [True, False]

    UNIQUE_PARAMS = list({'param_' + _ for __ in param_grid for _ in __})

    test_permutations = np.prod(list(map(len, (CV, SCORING, RETURN_TRAIN))))
    print(f'number of test permutations = {test_permutations}\n')

    ctr = 0
    for _cv in CV:
        for _scoring in SCORING:
            for return_train in RETURN_TRAIN:
                ctr += 1
                print(f'\033[92mRunning test permutation number {ctr} of {test_permutations}...\033[0m')
                # BUILD VERIFICATION STUFF #########################################################################
                COLUMN_CHECK = ['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time']
                COLUMN_CHECK += UNIQUE_PARAMS
                COLUMN_CHECK += ['params']
                for sub_scoring in _scoring:
                    suffix = 'score' if len(_scoring)==1 else sub_scoring
                    for split in range(_cv):
                        COLUMN_CHECK += [f'split{split}_test_{suffix}']
                    COLUMN_CHECK += [f'mean_test_{suffix}']
                    COLUMN_CHECK += [f'std_test_{suffix}']
                    COLUMN_CHECK += [f'rank_test_{suffix}']
                    if return_train:
                        for split in range(_cv):
                            COLUMN_CHECK += [f'split{split}_train_{suffix}']
                        COLUMN_CHECK += [f'mean_train_{suffix}']
                        COLUMN_CHECK += [f'std_train_{suffix}']

                # BUILD VERIFICATION STUFF #########################################################################

                # RUN cv_results_builder AND GET CHARACTERISTICS ###################################################
                cv_results_output = cv_results_builder(param_grid, _cv, _scoring, return_train)
                OUTPUT_COLUMNS = list(cv_results_output.keys())
                OUTPUT_LEN = len(cv_results_output['mean_fit_time'])
                # RUN cv_results_builder AND GET CHARACTERISTICS ###################################################

                # COMPARE OUTPUT TO CONTROLS ########################################################################
                for out_col in OUTPUT_COLUMNS:
                    if out_col not in COLUMN_CHECK:
                        raise Exception(f"\033[91m{out_col} is in OUTPUT_COLUMNS but not in COLUMN_CHECK\033[0m")
                for check_col in COLUMN_CHECK:
                    if check_col not in OUTPUT_COLUMNS:
                        raise Exception(f"\033[91m{check_col} is in COLUMN_CHECK but not in OUTPUT_COLUMNS\033[0m")

                output_len = len(cv_results_output['mean_fit_time'])
                if output_len != correct_cv_results_len:
                    raise Exception(f"\033[91moutput rows ({output_len}) does not equal expected rows ({correct_cv_results_len})\033[0m")

                del output_len

                print(f'\033[92mTrial {ctr} passed all tests\033[0m')

                # COMPARE OUTPUT TO CONTROLS ########################################################################

    # END TEST FOR cv_results_builder ######################################################################################






