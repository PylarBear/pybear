import numpy as n, pandas as p
from ML_PACKAGE.standard_configs import config_notification_templates as cnt



def powerball_rv_methods():
    return [
        'STANDARD'
    ]


def powerball_reference_vectors_config(REFERENCE_VECTORS, standard_config):

    rv_method = cnt.load_config_template(standard_config, 'REFERENCE VECTORS', powerball_rv_methods())

    if rv_method == 'STANDARD':
        pass

    return REFERENCE_VECTORS


