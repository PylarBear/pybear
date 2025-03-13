import numpy as n, pandas as p
from data_validation import validate_user_input as vui
from ML_PACKAGE.standard_configs import config_notification_templates as cnt


def LEFT1_TOP1_big_matrix_methods():
    return [
        'STANDARD',
        'INTERCEPT'
    ]


def LEFT1_TOP1_big_matrix_config(standard_config, BIG_MATRIX, BASE_BIG_MATRIX, RETAINED_ATTRIBUTES, interaction,
                                int_cutoff, intercept, things_in_intercept):

    bm_method = cnt.load_config_template(standard_config, 'BIG MATRIX', LEFT1_TOP1_big_matrix_methods())

    if bm_method == 'STANDARD':

        interaction = 'N'
        int_cutoff = 0
        intercept = 'N'
        things_in_intercept = 'N'
        RETAINED_ATTRIBUTES.clear()
        TARGET_VECTOR_CHOPPED = [[]]
        BIG_MATRIX = BASE_BIG_MATRIX

    elif bm_method == 'INTERCEPT':

        interaction = 'N'
        int_cutoff = 0
        intercept = 'Y'
        things_in_intercept = 'N'
        RETAINED_ATTRIBUTES.clear()
        TARGET_VECTOR_CHOPPED = [[]]
        BIG_MATRIX = BASE_BIG_MATRIX
        n.insert(BIG_MATRIX, len(BIG_MATRIX), [1 for _ in range(len(BIG_MATRIX[0]))] ,axis=0)


    return BIG_MATRIX, RETAINED_ATTRIBUTES, TARGET_VECTOR_CHOPPED, interaction, int_cutoff, intercept, \
           things_in_intercept
