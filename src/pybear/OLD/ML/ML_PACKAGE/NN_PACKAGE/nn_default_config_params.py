import numpy as n


def nn_default_config_params():

    ARRAY_OF_NODES = [[], []]
    NEURONS = [10, 1]
    nodes = 2
    node_seed = n.random.rand()
    activation_constant = 1
    aon_base_path = r''
    aon_filename = r''
    cost_fxn = 'L'
    SELECT_LINK_FXN = ['Logistic', 'Logistic']
    LIST_OF_NN_ELEMENTS = []
    OUTPUT_VECTOR = []
    batch_method = 'B'
    BATCH_SIZE = []
    gd_method = 'G'
    conv_method = 'A'
    lr_method = 'C'
    gd_iterations = 1000
    LEARNING_RATE = [[.0001 for __ in range(gd_iterations)], [.00001 for __ in range(gd_iterations)]]
    momentum_weight = 0.5
    conv_kill = 100
    pct_change = 0.1
    conv_end_method = 'KILL'
    rglztn_type = 'L2'
    rglztn_fctr = 0.5
    non_neg_coeffs = 'N'
    allow_summary_print = 'Y'
    summary_print_interval = 50
    iteration = 0




    return ARRAY_OF_NODES, NEURONS,  nodes, node_seed, activation_constant, aon_base_path, aon_filename, cost_fxn, \
        SELECT_LINK_FXN, LIST_OF_NN_ELEMENTS, OUTPUT_VECTOR, batch_method, BATCH_SIZE, gd_method, conv_method, \
        lr_method, gd_iterations, LEARNING_RATE, momentum_weight, conv_kill, pct_change, conv_end_method, \
        rglztn_type, rglztn_fctr, non_neg_coeffs, allow_summary_print, summary_print_interval, iteration




