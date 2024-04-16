from ML_PACKAGE.standard_configs import config_notification_templates as cnt


def powerball_NN_methods():
    return [
        'STANDARD'
    ]


def powerball_NN_config(standard_config, ___):

    nn_method = cnt.load_config_template(standard_config, 'NN', powerball_NN_methods())

    if nn_method == 'STANDARD':

        ___ = ___

    return ___














