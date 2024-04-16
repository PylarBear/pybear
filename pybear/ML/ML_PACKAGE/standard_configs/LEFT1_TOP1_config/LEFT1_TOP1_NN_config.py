from ML_PACKAGE.standard_configs import config_notification_templates as cnt

# CALLED ONLY HERE
def LEFT1_TOP1_NN_methods():
    return [
        'STANDARD'
    ]

# CALLED BY standard_configs.standard_configs.NN_standard_configs() WHICH IS HASHED OUT
def LEFT1_TOP1_NN_config(standard_config, ___):

    nn_method = cnt.load_config_template(standard_config, 'NN', LEFT1_TOP1_NN_methods())

    if nn_method == 'STANDARD':

        ___ = ___

    return ___




# 2-7-22 RELIC -- MERGE WITH LEFT1_TOP1_NN_methods()
def LEFT1_TOP1_GD_methods():
    return [
        'STANDARD'
    ]

# CALLED BY standard_configs.standard_configs.GD_standard_configs()
def LEFT1_TOP1_GD_config(standard_config, cost_fxn, non_neg_coeffs):

    gd_method = cnt.load_config_template(standard_config, 'GD', LEFT1_TOP1_GD_methods())

    if gd_method == 'STANDARD':

        cost_fxn = 'L'
        non_neg_coeffs = 'N'

    return cost_fxn, non_neg_coeffs












