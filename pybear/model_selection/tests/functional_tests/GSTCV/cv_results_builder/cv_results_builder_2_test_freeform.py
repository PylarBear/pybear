# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import numpy as np

from model_selection.GSTCV._cv_results_builder._cv_results_builder import cv_results_builder




# TEST FOR cv_results_builder ######################################################################################
print(f'RUNNING cv_results_builder TEST')

param_grid = [
    {'kernel': ['rbf'], 'gamma': [0.1, 0.2], 'test_param': [1, 2, 3]},
    {'kernel': ['poly'], 'degree': [2, 3], 'test_param': [1, 2, 3]},
]

correct_cv_results_len = np.sum(list(map(
    np.prod,
    [[len(_) for _ in __] for __ in map(dict.values, param_grid)]
)))

CV = [3, 4, 5]

SCORING = [['accuracy'], ['accuracy', 'balanced_accuracy']]

RETURN_TRAIN = [True, False]

UNIQUE_PARAMS = list({'param_' + _ for __ in param_grid for _ in __})  # notice set!

test_permutations = np.prod(list(map(len, (CV, SCORING, RETURN_TRAIN))))
print(f'number of test permutations = {test_permutations}\n')

ctr = 0
for _cv in CV:
    for _scoring in SCORING:
        for return_train in RETURN_TRAIN:
            ctr += 1
            print(f'\033[92mRunning test permutation number {ctr} of {test_permutations}...\033[0m')
            # BUILD VERIFICATION STUFF #################################
            COLUMN_CHECK = ['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time']
            COLUMN_CHECK += UNIQUE_PARAMS
            COLUMN_CHECK += ['params']
            for sub_scoring in _scoring:
                COLUMN_CHECK += ['best_threshold' + ('' if len(_scoring) == 1 else f'_{sub_scoring}')]

                suffix = 'score' if len(_scoring) == 1 else sub_scoring
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

            # END BUILD VERIFICATION STUFF #############################

            # RUN cv_results_builder AND GET CHARACTERISTICS ###########
            cv_results_output = cv_results_builder(param_grid, _cv, _scoring, return_train)[0]
            OUTPUT_COLUMNS = list(cv_results_output.keys())
            OUTPUT_LEN = len(cv_results_output['mean_fit_time'])
            # RUN cv_results_builder AND GET CHARACTERISTICS ###########

            # COMPARE OUTPUT TO CONTROLS ###############################
            for out_col in OUTPUT_COLUMNS:
                if out_col not in COLUMN_CHECK:
                    print(f'OUTPUT_COLUMNS = ')
                    print(OUTPUT_COLUMNS)
                    print()
                    print(f'COLUMN_CHECK = ')
                    print(COLUMN_CHECK)
                    raise Exception(
                        f"\033[91m{out_col} is in OUTPUT_COLUMNS but not in COLUMN_CHECK\033[0m")

            for check_col in COLUMN_CHECK:
                if check_col not in OUTPUT_COLUMNS:
                    raise Exception(
                        f"\033[91m{check_col} is in COLUMN_CHECK but not in OUTPUT_COLUMNS\033[0m")

            if OUTPUT_LEN != correct_cv_results_len:
                raise Exception(
                    f"\033[91moutput rows ({output_len}) does not equal expected rows ({correct_cv_results_len})\033[0m")

            print(f'\033[92mTrial {ctr} passed all tests\033[0m')

            # COMPARE OUTPUT TO CONTROLS ###############################

# END TEST FOR cv_results_builder ######################################





















