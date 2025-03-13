import numpy as n
from ML_PACKAGE.standard_configs import config_notification_templates as cnt
from ML_PACKAGE.TEST_CONFIG_RUN import generic_nn_test_run as gntr


def AA_test_cases_calc_methods():
    return [
        'GENERIC NN TEST CALC'
    ]


# CALLED BY standard_configs.test_cases_standard_configs()
def AA_test_cases_calc(standard_config, TEST_MATRIX, BIG_MATRIX, ARRAY_OF_NODES,
                       SELECT_LINK_FXN, RETAINED_ATTRIBUTES, activation_constant):

    tc_method = cnt.load_config_template(standard_config, 'TEST CASES', AA_test_cases_calc_methods())

    if tc_method == 'GENERIC NN TEST CALC':
        CSUTM_DF = gntr.test_cases_calculator(standard_config, tc_method, TEST_MATRIX, ARRAY_OF_NODES, SELECT_LINK_FXN,
                      RETAINED_ATTRIBUTES, activation_constant)

    return CSUTM_DF











if __name__ == '__main__':
    from ML_PACKAGE.NN_PACKAGE.link_functions import link_fxns as lf
    SELECT_LINK_FXN = []
    NEURONS = []

    standard_config = 'AA'
    tc_method = ''
    TEST_OBJECT = n.array([[1,3,2],[1,2,3],[2,1,6]], dtype=object)
    ARRAY_OF_NODES = n.array([
                         n.array([[.1,.1,.1],[.1,.1,.1],[.1,.1,.1]], dtype=float),
                         n.array([[.1,.1,.1]], dtype=float)
                        ], dtype=object)
    from ML_PACKAGE.NN_PACKAGE.link_functions import select_link_fxn as lf
    SELECT_LINK_FXN = lf.select_link_fxn(ARRAY_OF_NODES, lf.define_links(), SELECT_LINK_FXN, NEURONS)
    RETAINED_ATTRIBUTES = n.array([
                        ['TEST', 'TEST', 'TEST'],
                        ['A', 'B', 'C']
                        ], dtype=object)
    activation_constant = 1

    CSUTM_DF = generic_nn_test_run(standard_config, tc_method, TEST_OBJECT, ARRAY_OF_NODES, SELECT_LINK_FXN,
                          RETAINED_ATTRIBUTES, activation_constant)

    print(CSUTM_DF)







































