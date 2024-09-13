import numpy as n
from ML_PACKAGE.standard_configs import config_notification_templates as cnt


def LEFT1_TOP1_rv_methods():
    return [
        'STANDARD'
    ]


def LEFT1_TOP1_reference_vectors_config(REFERENCE_VECTORS, standard_config):

    rv_method = cnt.load_config_template(standard_config, 'REFERENCE VECTORS', LEFT1_TOP1_rv_methods())

    if rv_method == 'STANDARD':
        pass


    return REFERENCE_VECTORS


