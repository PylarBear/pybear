import numpy as n


def svm_default_config_params():

    margin_type = 'SOFT'
    C = float('inf')
    cost_fxn = 'H'
    kernel_fxn = 'LINEAR'
    constant = 0
    exponent = 1
    sigma = 1
    K = [[]]
    ALPHAS = []
    b = 0
    alpha_seed = 0
    alpha_selection_alg = 'SMO'
    max_passes = 20000
    tol = 0.001
    svm_conv_kill = 500
    svm_pct_change = 0
    svm_conv_end_method = 'KILL'
    alpha_selection_alg = 'SMO'
    SMO_a2_selection_method = 'RANDOM'

    return margin_type, C, cost_fxn, kernel_fxn, constant, exponent, sigma, K, ALPHAS, b, alpha_seed, alpha_selection_alg, \
           max_passes, tol, svm_conv_kill, svm_pct_change, svm_conv_end_method, alpha_selection_alg, SMO_a2_selection_method