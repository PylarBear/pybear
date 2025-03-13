import numpy as n, pandas as p
from data_validation import validate_user_input as vui
from ML_PACKAGE.standard_configs import config_notification_templates as cnt
from ML_PACKAGE.DATA_PREP_IN_SITU_PACKAGE.big_matrix.setup_change_matrix import setup_change_matrix as scm


def AA_big_matrix_methods():
    return [
        'STANDARD'
    ]


def AA_big_matrix_config(standard_config, BIG_MATRIX, BASE_BIG_MATRIX, RETAINED_ATTRIBUTES, COLIN_CHOPPED, MKT, \
                         interaction, int_cutoff, intercept, things_in_intercept):

    bm_method = cnt.load_config_template(standard_config, 'BIG MATRIX', AA_big_matrix_methods())

    if bm_method == 'STANDARD':
        # TRIM HEADER COLUMNS AND ROWS FROM BASE_BIG_MATRIX, AUGMENT W USER-CHOICE
        # INTERACTIONS & INTERCEPT#########
        print(f'Loading AA BIG MATRIX config...\n')

        interaction, int_cutoff, intercept, BIG_MATRIX, RETAINED_ATTRIBUTES, TARGET_VECTOR_CHOPPED, \
        APPDATE_VECTOR, COLIN_CHOPPED, things_in_intercept = \
            scm.setup_change_matrix(BASE_BIG_MATRIX, interaction, int_cutoff, intercept,
                                    RETAINED_ATTRIBUTES, COLIN_CHOPPED, MKT, things_in_intercept)



    return BIG_MATRIX, RETAINED_ATTRIBUTES, TARGET_VECTOR_CHOPPED, APPDATE_VECTOR, COLIN_CHOPPED, \
           interaction, int_cutoff, intercept, things_in_intercept

