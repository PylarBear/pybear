import time
import numpy as np
from ML_PACKAGE.MLREGRESSION.misc_test import coeff_calculator as cc, r2_calculator as rc

# 6/1/23 THIS IS A TEST FOR COEFF AND R2 CALCULATOR FUNCTIONS USED IN TEST FOR THE ACCURACY OF MLRegression()
# SO THIS IS A TEST FOR A TEST

# REFEREE VALUES WERE OBTAINED VIA EXCEL, ONLY FOR THE NON-RIDGE CASE

# TEST CODE, cc, AND cr ARE TESTED & GOOD AS OF 6/1/23

GIVEN_TARGET = np.array([1.5, 3, 3, 3, 4, 3, 3.5, 3, 4], dtype=np.float64).reshape((-1, 1))

review_taste = np.array([1.5, 3, 3, 3, 4.5, 3.5, 4, 3.5, 4], dtype=np.float64)
review_palate = np.array([1.5, 3, 3, 2.5, 4, 3, 4, 2, 3.5], dtype=np.float64)
review_aroma = np.array([2, 2.5, 2.5, 3, 4.5, 3.5, 3.5, 2.5, 3], dtype=np.float64)
review_appearance = np.array([2.5, 3, 3, 3.5, 4, 3.5, 3.5, 3.5, 3.5], dtype=np.float64)

BASE_DATA = np.vstack((review_taste, review_palate, review_aroma, review_appearance)).transpose()
HEADER = ['review_taste', 'review_palate', 'review_aroma', 'review_appearance']

rglztn_fctr = 0

# TEST INDIVIDUAL COLUMNS FOR INTERCEPT CASE ##################################################################
    # ALSO USE np.corrcoeff TO REFEREE IN THIS CASE

print(f'\nRunning INDIVIDUAL COLUMNS FOR INTERCEPT CASE...')

# [COEFF, INTERCEPT] AND RSQ FOR HEADER IN ORDER
COEFF_KEY = [[.819444,.379630],[.752427, .895631],[0.722222, .944444],[1.444444, -1.703704]]
RSQ_KEY = [.917985, .738141, .534810, .713080]

for idx, COLUMN in enumerate(BASE_DATA.transpose()):
    print(f'   Running {HEADER[idx]}...')

    COLUMN = np.insert(COLUMN.reshape((1,-1)), 1, 1, axis=0).transpose()  # APPEND ONES TO COLUMN

    COEFFS_AS_ROW = cc.exp_coeff_calc(COLUMN, GIVEN_TARGET, rglztn_fctr)
    ROUNDED_COEFFS_AS_ROW = np.round(COEFFS_AS_ROW, 6)
    ROUNDED_COEFFS_AS_COLUMN = ROUNDED_COEFFS_AS_ROW.transpose()
    if not np.array_equiv(COEFF_KEY[idx], ROUNDED_COEFFS_AS_COLUMN):
        print(f'\033[91mEXP = {COEFF_KEY[idx]}')
        print(f'ACT = {ROUNDED_COEFFS_AS_COLUMN}')
        time.sleep(0.5)
        raise Exception(f'*** INCONGRUENT COEFFS FOR {HEADER[idx]} WITH INTERCEPT ***')

    exp_r2 = rc.exp_r2_calc(COLUMN, GIVEN_TARGET, COEFFS_AS_ROW, has_intercept=True)
    exp_r2 = round(exp_r2, 6)

    if exp_r2 != RSQ_KEY[idx]:
        print(f'\033[91mEXP = {RSQ_KEY[idx]}')
        print(f'ACT = {exp_r2}')
        time.sleep(0.5)
        raise Exception(f'*** UNEQUAL RSQ FOR {HEADER[idx]} WITH INTERCEPT ***')

    PREDICTED_AS_COLUMN = np.matmul(COLUMN.astype(np.float64), COEFFS_AS_ROW.astype(np.float64)).astype(np.float64).transpose()

    np_rsq = np.round(np.corrcoef(PREDICTED_AS_COLUMN, GIVEN_TARGET.transpose())[0][1]**2, 6)
    if exp_r2 != np_rsq:
        print(f'\033[91mEXP = {np_rsq}')
        print(f'ACT = {exp_r2}')
        time.sleep(0.5)
        raise Exception(f'*** UNEQUAL RSQ FOR {HEADER[idx]} WITH INTERCEPT AGAINST np.corrcoeff ***')

print(f'\n\033[92m*** INDIVIDUAL COLUMNS FOR INTERCEPT CASE PASSED ***\033[0m')


# END TEST INDIVIDUAL COLUMNS FOR INTERCEPT CASE ##################################################################



# TEST CUMUL COLUMNS FOR INTERCEPT CASE ##################################################################
    # ALSO USE np.corrcoeff TO REFEREE IN THIS CASE

print(f'\nRunning CUMUL COLUMNS FOR INTERCEPT CASE...')

# [COEFF, INTERCEPT] AND RSQ FOR HEADER IN ORDER
COEFF_KEY = [[.379630, .819444], [.344839, .685578, .163363],[.481520, .794661,.236140,-.238193],[-1.709091, .163636,.618182,-.636364,1.309091]]
RSQ_KEY = [.917985, .928282, .945494,.964787]

for idx, num_cols in enumerate(range(1,5)):
    print(f'   Running {num_cols} CUMUL COLUMNS...')

    COLUMNS = BASE_DATA[..., :num_cols].transpose()

    COLUMNS = np.insert(COLUMNS, 0, 1, axis=0).transpose()  # APPEND ONES TO COLUMN

    COEFFS_AS_ROW = cc.exp_coeff_calc(COLUMNS, GIVEN_TARGET, rglztn_fctr)
    ROUNDED_COEFFS_AS_ROW = np.round(COEFFS_AS_ROW, 6)
    ROUNDED_COEFFS_AS_COLUMN = ROUNDED_COEFFS_AS_ROW.transpose()
    if not np.array_equiv(COEFF_KEY[idx], ROUNDED_COEFFS_AS_COLUMN):
        print(f'\033[91mEXP = {COEFF_KEY[idx]}')
        print(f'ACT = {ROUNDED_COEFFS_AS_COLUMN}')
        time.sleep(0.5)
        raise Exception(f'*** INCONGRUENT COEFFS FOR {num_cols} CUMUL COLUMNS WITH INTERCEPT ***')

    exp_r2 = rc.exp_r2_calc(COLUMNS, GIVEN_TARGET, COEFFS_AS_ROW, has_intercept=True)
    exp_r2 = round(exp_r2, 6)

    if exp_r2 != RSQ_KEY[idx]:
        print(f'\033[91mEXP = {RSQ_KEY[idx]}')
        print(f'ACT = {exp_r2}')
        time.sleep(0.5)
        raise Exception(f'*** UNEQUAL RSQ FOR {num_cols} CUMUL COLUMNS WITH INTERCEPT ***')

    PREDICTED_AS_COLUMN = np.matmul(COLUMNS.astype(np.float64), COEFFS_AS_ROW.astype(np.float64)).astype(np.float64).transpose()

    np_rsq = np.round(np.corrcoef(PREDICTED_AS_COLUMN, GIVEN_TARGET.transpose())[0][1]**2, 6)
    if exp_r2 != np_rsq:
        print(f'\033[91mEXP = {np_rsq}')
        print(f'ACT = {exp_r2}')
        time.sleep(0.5)
        raise Exception(f'*** UNEQUAL RSQ FOR {num_cols} CUMUL COLUMNS WITH INTERCEPT AGAINST np.corrcoeff ***')

print(f'\n\033[92m*** CUMUL COLUMNS FOR INTERCEPT CASE PASSED ***\033[0m')


# END TEST CUMUL COLUMNS FOR INTERCEPT CASE ############################################################################




# TEST INDIVIDUAL COLUMNS FOR NO INTERCEPT CASE ##################################################################
    # DONT USE np.corrcoeff TO REFEREE IN THIS CASE

print(f'\nRunning INDIVIDUAL COLUMNS FOR THE NO INTERCEPT CASE...')

COEFF_KEY = [.926887, 1.035821, 1.020468, .940887]
RSQ_KEY = [.995264, .982049, .973069, .982018]

for idx, COLUMN in enumerate(BASE_DATA.transpose()):
    print(f'   Running {HEADER[idx]}...')

    COLUMN = COLUMN.reshape((1, -1)).transpose()

    COEFFS_AS_ROW = cc.exp_coeff_calc(COLUMN, GIVEN_TARGET, rglztn_fctr)
    ROUNDED_COEFFS_AS_ROW = np.round(COEFFS_AS_ROW, 6)
    ROUNDED_COEFFS_AS_COLUMN = ROUNDED_COEFFS_AS_ROW.transpose()
    if not np.array_equiv(COEFF_KEY[idx], ROUNDED_COEFFS_AS_COLUMN):
        print(f'\033[91mEXP = {COEFF_KEY[idx]}')
        print(f'ACT = {ROUNDED_COEFFS_AS_COLUMN}')
        time.sleep(0.5)
        raise Exception(f'*** INCONGRUENT COEFFS FOR {HEADER[idx]} WITHOUT INTERCEPT ***')

    exp_r2 = rc.exp_r2_calc(COLUMN, GIVEN_TARGET, COEFFS_AS_ROW, has_intercept=False)
    exp_r2 = round(exp_r2, 6)

    if exp_r2 != RSQ_KEY[idx]:
        print(f'\033[91mEXP = {RSQ_KEY[idx]}')
        print(f'ACT = {exp_r2}')
        time.sleep(0.5)
        raise Exception(f'*** UNEQUAL RSQ FOR {HEADER[idx]} WITHOUT INTERCEPT ***')

print(f'\n\033[92m*** INDIVIDUAL COLUMNS FOR THE NO INTERCEPT CASE PASSED ***\033[0m')


# END TEST INDIVIDUAL COLUMNS FOR NO INTERCEPT CASE ##################################################################



# TEST CUMUL COLUMNS FOR NO INTERCEPT CASE ##################################################################
    # DONT USE np.corrcoeff TO REFEREE IN THIS CASE

print(f'\nRunning CUMUL COLUMNS FOR THE NO INTERCEPT CASE...')

COEFF_KEY = [[.926887],[.763999, .185160],[.849578,.234869,-.145431],[.629858,.331606,-.350014,.326536]]
RSQ_KEY = [.995264, .995907, .996253, .997769]

for idx, num_cols in enumerate(range(1,5)):
    print(f'   Running {num_cols} CUMUL COLUMNS...')

    COLUMNS = BASE_DATA[..., :num_cols]

    COEFFS_AS_ROW = cc.exp_coeff_calc(COLUMNS, GIVEN_TARGET, rglztn_fctr)
    ROUNDED_COEFFS_AS_ROW = np.round(COEFFS_AS_ROW, 6)
    ROUNDED_COEFFS_AS_COLUMN = ROUNDED_COEFFS_AS_ROW.transpose()
    if not np.array_equiv(COEFF_KEY[idx], ROUNDED_COEFFS_AS_COLUMN):
        print(f'\033[91mEXP = {COEFF_KEY[idx]}')
        print(f'ACT = {ROUNDED_COEFFS_AS_COLUMN}')
        time.sleep(0.5)
        raise Exception(f'*** INCONGRUENT COEFFS FOR {num_cols} CUMUL COLUMNS WITHOUT INTERCEPT ***')

    exp_r2 = rc.exp_r2_calc(COLUMNS, GIVEN_TARGET, COEFFS_AS_ROW, has_intercept=False)
    exp_r2 = round(exp_r2, 6)

    if exp_r2 != RSQ_KEY[idx]:
        print(f'\033[91mEXP = {RSQ_KEY[idx]}')
        print(f'ACT = {exp_r2}')
        time.sleep(0.5)
        raise Exception(f'*** UNEQUAL RSQ FOR {num_cols} CUMUL COLUMNS WITHOUT INTERCEPT ***')

print(f'\n\033[92m*** CUMUL COLUMNS FOR THE NO INTERCEPT CASE PASSED ***')


# END TEST CUMUL COLUMNS FOR NO INTERCEPT CASE ##################################################################







