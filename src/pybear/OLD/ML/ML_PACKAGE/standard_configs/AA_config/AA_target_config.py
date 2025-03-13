from copy import deepcopy
from ML_PACKAGE.standard_configs import config_notification_templates as cnt
from ML_PACKAGE.GENERIC_PRINT import print_object_create_success as pocs

'''
    FOR TARGET CONFIG, NEED TO SPECIFY
    1) RAW_TARGET_VECTOR (IF DOING A CUSTOM RTV, AFTER WHICH CUSTOM RTV GETS STANDARD PrerunRun TREATMENT)
    2) RAW_TARGET_VECTOR_HEADER (IF DOING A CUSTOM RTV HEADER, AFTER WHICH CUSTOM RTV GETS STANDARD PrerunRun TREATMENT)
    3) RAW_TARGET_VECTOR (IF DOING A CUSTOM TV, AFTER WHICH CUSTOM TV JUST PASSES THRU PrerunRun W/O TREATMENT)
    4) RAW_TARGET_VECTOR (IF DOING A CUSTOM TV HEADER, AFTER WHICH CUSTOM TV HEADER JUST PASSES THRU PrerunRun W/O TREAMMENT)
    5) SPLIT METHOD
    6) NUMBER OF LABELS
    7) LABEL RULES    
    8) EVENT VALUE
    9) NON-EVENT VALUE
    10) COLUMNS TO KEEP FROM TARGET_VECTOR_SOURCE (TO TURN IT INTO RAW_TARGET_VECTOR)
'''

# 1-10-22 NOTES --- RAW_TARGET_SOURCE = AS READ FROM FILE, RAW_TARGET_VECTOR = AFTER MODS TO SOURCE (COLUMN CHOPS)

# UNIQUE TO THIS MODULE
def allowed_str():
    return 'ADEJKLMNOPQRSTUVWXYZ'


#CALLED BY standard_configs.AA_config.AA_target_config.AA_target_configs()
def AA_target_methods():
    return ['BINARY - ALL HITS (NON-OXYZ)',
            'BINARY - JK HITS ONLY',
            'SOFTMAX',
            'SVM - ALL HITS (NON-OXYZ)',
            'SVM - JK HITS ONLY'
            ]

#CALLED BY standard_configs.standard_configs.TARGET_VECTOR_standard_configs()
def AA_target_configs(standard_config, target_config, RAW_TARGET_SOURCE, RAW_TARGET_SOURCE_HEADER, TARGET_VECTOR,
                      TARGET_VECTOR_HEADER):

    # CHECK STATUS CODES
    for status in RAW_TARGET_SOURCE[0]:
        if status not in allowed_str():
            raise ValueError(f'Program terminated for bad status code in raw data.')
    print('Status codes are OK.')

    target_method = cnt.load_config_template(standard_config, 'TARGET VECTOR', AA_target_methods())

    if target_method == 'SOFTMAX':
        RAW_TARGET_VECTOR = deepcopy(RAW_TARGET_SOURCE)
        RAW_TARGET_VECTOR_HEADER = deepcopy(RAW_TARGET_SOURCE_HEADER)
        split_method = 'c2c'
        LABEL_RULES = [
            ['O', 'X', 'Y', 'Z'],
            ['J', 'K', 'E', 'P', 'U', 'V'],
            ['C']
        ]
        number_of_labels = 3
        event_value = 1
        negative_value = 0
        keep_columns = []

    elif target_method == 'SVM - ALL HITS (NON-OXYZ)':

        RAW_TARGET_VECTOR = deepcopy(RAW_TARGET_SOURCE)
        RAW_TARGET_VECTOR_HEADER = deepcopy(RAW_TARGET_SOURCE_HEADER)
        split_method = 'c2c'
        LABEL_RULES = [
            [_ for _ in allowed_str() if _ not in 'OXYZ']
        ]
        number_of_labels = 1
        event_value = 1
        negative_value = -1
        keep_columns = []

    elif target_method == 'SVM - JK HITS ONLY':

        RAW_TARGET_VECTOR = deepcopy(RAW_TARGET_SOURCE)
        RAW_TARGET_VECTOR_HEADER = deepcopy(RAW_TARGET_SOURCE_HEADER)
        split_method = 'c2c'
        LABEL_RULES = [
            ['J', 'K']
        ]
        number_of_labels = 1
        event_value = 1
        negative_value = -1
        keep_columns = []

    elif target_method == 'BINARY - ALL HITS (NON-OXYZ)':

        RAW_TARGET_VECTOR = deepcopy(RAW_TARGET_SOURCE)
        RAW_TARGET_VECTOR_HEADER = deepcopy(RAW_TARGET_SOURCE_HEADER)
        split_method = 'c2c'
        LABEL_RULES = [
            [_ for _ in allowed_str() if _ not in 'OXYZ']
        ]
        number_of_labels = 1
        event_value = 1
        negative_value = 0
        keep_columns = []

    elif target_method == 'BINARY - JK HITS ONLY':

        RAW_TARGET_VECTOR = deepcopy(RAW_TARGET_SOURCE)
        RAW_TARGET_VECTOR_HEADER = deepcopy(RAW_TARGET_SOURCE_HEADER)
        split_method = 'c2c'
        LABEL_RULES = [
            ['J', 'K']
        ]
        number_of_labels = 1
        event_value = 1
        negative_value = 0
        keep_columns = []

    else:
        raise ValueError(f'\nTHERE IS A MISMATCH IN THE TARGET VECTOR STANDARD CONFIG NOMENCLATURE AND THE '
                         f'STANDARD CONFIG NAMES BEING USED IN THE AA_target_configs CODE.\n')

    pocs.print_object_create_success(standard_config, 'TARGET CATEGORY SPLITTER RULES')

    return RAW_TARGET_VECTOR, RAW_TARGET_VECTOR_HEADER, TARGET_VECTOR, TARGET_VECTOR_HEADER, LABEL_RULES, \
           number_of_labels, split_method, event_value, negative_value, keep_columns









if __name__ == '__main__':
    standard_config = 'AA'
    target_config = 'I'
    RAW_TARGET_SOURCE = ['X','O','X','O','O','P','O']
    RAW_TARGET_SOURCE_HEADER = [['STATUS']]
    TARGET_VECTOR = []
    TARGET_VECTOR_HEADER = [[]]

    AA_target_configs(standard_config, target_config, RAW_TARGET_SOURCE, RAW_TARGET_SOURCE_HEADER, TARGET_VECTOR,
                      TARGET_VECTOR_HEADER)

