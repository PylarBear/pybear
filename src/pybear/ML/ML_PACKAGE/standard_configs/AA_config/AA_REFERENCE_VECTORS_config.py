from ML_PACKAGE.standard_configs.AA_config import appdate_config as ac, rowid_config as rc
from ML_PACKAGE.standard_configs import config_notification_templates as cnt


def AA_rv_methods():
    return [
        'STANDARD'
    ]


def AA_reference_vectors_config(REFERENCE_VECTORS, standard_config):

    rv_method = cnt.load_config_template(standard_config, 'REFERENCE VECTORS', AA_rv_methods())

    if rv_method == 'STANDARD':
        pass
        # REFERENCE_VECTORS.clear()
        # REFERENCE_VECTORS.append(ac.appdate_config(DATA_DF))
        # REFERENCE_VECTORS.append(rc.rowid_config(DATA_DF))


    return REFERENCE_VECTORS





