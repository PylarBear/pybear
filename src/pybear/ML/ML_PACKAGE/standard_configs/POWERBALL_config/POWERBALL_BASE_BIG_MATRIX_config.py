import numpy as n, pandas as p
from data_validation import validate_user_input as vui
from ML_PACKAGE.standard_configs import config_notification_templates as cnt


def powerball_base_big_matrix_methods():
    return [
        'STANDARD',
        'DATE-ONLY'
    ]


def powerball_base_big_matrix_config(DATA_DF, BASE_BIG_MATRIX, standard_config):

    bbm_method = cnt.load_config_template(standard_config, 'BASE BIG MATRIX', powerball_base_big_matrix_methods())

    if bbm_method == 'STANDARD':

        number_of_draws = vui.validate_user_int(f'Enter number of draws > ', min=1, max=len(DATA_DF['NUMBER1']))

        NUMPY = n.array(DATA_DF, dtype=int)

        BASE_BIG_MATRIX = [[elmt for elmt in x] for x in BASE_BIG_MATRIX]
        BASE_BIG_MATRIX.clear()

        for row_idx in range(1,len(NUMPY)-number_of_draws+1):
            BASE_BIG_MATRIX.append([])
            for row_idx2 in range(0,number_of_draws):
                BASE_BIG_MATRIX[-1] += [DRAW for DRAW in NUMPY[row_idx+row_idx2]]

        BASE_BIG_MATRIX = n.transpose(BASE_BIG_MATRIX)  #GETS IT READY FOR NUMPY MATMUL

    elif bbm_method == 'DATE-ONLY':

        number_of_draws = vui.validate_user_int(f'Enter number of draws > ', min=1, max=len(DATA_DF['NUMBER1']))

        NUMPY = n.array(DATA_DF[['DD','MM','YYYY']], dtype=int)

        BASE_BIG_MATRIX = [[elmt for elmt in x] for x in BASE_BIG_MATRIX]
        BASE_BIG_MATRIX.clear()

        for row_idx in range(1,len(NUMPY)-number_of_draws+1):
            BASE_BIG_MATRIX.append([])
            for row_idx2 in range(0,number_of_draws):
                BASE_BIG_MATRIX[-1] += [DRAW for DRAW in NUMPY[row_idx+row_idx2]]

        BASE_BIG_MATRIX = n.transpose(BASE_BIG_MATRIX)  #GETS IT READY FOR NUMPY MATMUL

    return BASE_BIG_MATRIX



