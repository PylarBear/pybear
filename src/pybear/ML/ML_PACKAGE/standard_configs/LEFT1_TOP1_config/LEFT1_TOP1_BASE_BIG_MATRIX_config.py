import numpy as n, pandas as p
from data_validation import validate_user_input as vui
from ML_PACKAGE.standard_configs import config_notification_templates as cnt


def LEFT1_TOP1_base_big_matrix_methods():
    return [
        'STANDARD'
    ]


def LEFT1_TOP1_base_big_matrix_config(DATA_DF, BASE_BIG_MATRIX, standard_config):

    bbm_method = cnt.load_config_template(standard_config, 'BASE BIG MATRIX', LEFT1_TOP1_base_big_matrix_methods())

    if bbm_method == 'STANDARD':

        NUMPY = n.array(DATA_DF)

        BASE_BIG_MATRIX = n.transpose(NUMPY)  #GETS IT READY FOR NUMPY MATMUL


    return BASE_BIG_MATRIX



