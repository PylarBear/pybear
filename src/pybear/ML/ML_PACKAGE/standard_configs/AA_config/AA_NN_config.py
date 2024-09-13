from ML_PACKAGE.standard_configs import config_notification_templates as cnt


def AA_NN_methods():
    return [
        'STANDARD'
    ]


def AA_NN_config(standard_config):

    gd_method = cnt.load_config_template(standard_config, 'NN', AA_NN_methods())

    if gd_method == 'STANDARD':

        ARRAY_OF_NODES = [[], []]
        # NEURONS = [3, 3, 1]
        NEURONS = [10, 1]
        # nodes = 3
        nodes = 2
        node_seed = 0
        activation_constant = 1
        aon_base_path = r''
        aon_filename = r''
        cost_fxn = 'L'
        # SELECT_LINK_FXN = ['Logistic', 'Logistic', 'Logistic']
        SELECT_LINK_FXN = ['Logistic', 'Logistic']
        LIST_OF_NN_ELEMENTS = []
        gd_iterations = 10000
        batch_method = 'B'
        gd_method = 'G'
        conv_method = 'A'
        lr_method = 'C'
        LEARNING_RATE = [[.1/(.998**(__+1)) for __ in range(gd_iterations)], [.01/(.998**(__+1)) for __ in range(gd_iterations)]]
        momentum_weight = 0.5
        rglztn_type = 'L2'
        rglztn_fctr = 0.5
        conv_kill = 1000
        pct_change = 0.1
        conv_end_method = 'KILL'
        non_neg_coeffs = 'N'
        allow_summary_print = 'Y'
        summary_print_interval = 100
        iteration = 0

    return ARRAY_OF_NODES, NEURONS, nodes, node_seed, activation_constant, aon_base_path, aon_filename, cost_fxn, \
           SELECT_LINK_FXN, LIST_OF_NN_ELEMENTS, gd_iterations, batch_method, gd_method, conv_method, \
           lr_method, LEARNING_RATE, momentum_weight, rglztn_type, rglztn_fctr, conv_kill, pct_change, conv_end_method, \
           non_neg_coeffs, allow_summary_print, summary_print_interval, iteration

















