import numpy as n, pandas as p
from ML_PACKAGE.standard_configs import config_notification_templates as cnt
from data_validation import validate_user_input as vui
from NN_PACKAGE.gd_run import output_vector_calc as ovc


def powerball_test_cases_calc_methods():
    return [
        'STANDARD',
    ]


# CALLED BY standard_configs.test_cases_standard_configs()
def powerball_test_cases_calc(standard_config, TEST_MATRIX, ARRAY_OF_NODES,
                              RETAINED_ATTRIBUTES, SELECT_LINK_FXN, activation_constant):

    tc_method = cnt.load_config_template(standard_config, 'TEST CASES', powerball_test_cases_calc_methods())

    if tc_method == 'STANDARD':

        OUTPUT_VECTOR = ovc.output_vector_calc(TEST_MATRIX, ARRAY_OF_NODES, SELECT_LINK_FXN, [[]], activation_constant)

        print(f'Resulting OUTPUT VECTOR for TEST MATRIX is:')
        [print(f'{x+1}) {OUTPUT_VECTOR[x][0]}') for x in range(len(OUTPUT_VECTOR))]

        return OUTPUT_VECTOR










