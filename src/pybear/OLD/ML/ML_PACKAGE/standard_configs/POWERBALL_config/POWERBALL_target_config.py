import numpy as n
from ML_PACKAGE.standard_configs import config_notification_templates as cnt
from data_validation import validate_user_input as vui


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


def powerball_target_methods():
    return ['STANDARD',
            'DRAW-ONLY',
            'SOFTMAX_POWERBALL_ONLY'
            ]


#CALLED BY TARGET_VECTOR_standard_configs
def powerball_target_configs(standard_config, target_config, RAW_TARGET_SOURCE, RAW_TARGET_SOURCE_HEADER, TARGET_VECTOR,
                             TARGET_VECTOR_HEADER):

    target_method = cnt.load_config_template(standard_config, 'TARGET VECTOR', powerball_target_methods())

    if target_method == 'STANDARD':

        TARGET_VECTOR.clear()
        NUMPY = n.array(RAW_TARGET_SOURCE, dtype=int)
        number_of_draws = vui.validate_user_int('\nEnter number of draws (must be the same as BIG MATRIX) > ', min=1, max=len(NUMPY))
        for row_idx in range(0, len(NUMPY) - number_of_draws):
            TARGET_VECTOR.append([x for x in NUMPY[row_idx]])

        number_of_labels = len(TARGET_VECTOR[0])  #USING MULTI-OUT, # OUTPUT VECTORS MUST = LEN OF TARGET EXAMPLES

    elif target_method == 'DRAW-ONLY':

        TARGET_VECTOR.clear()
        DRAW_ONLY = RAW_TARGET_SOURCE.drop(columns=['DD','MM','YYYY'])
        NUMPY = n.array(DRAW_ONLY, dtype=int)
        number_of_draws = vui.validate_user_int('\nEnter number of draws (must be the same as BIG MATRIX > ', min=1, max=len(NUMPY))
        for row_idx in range(0, len(NUMPY) - number_of_draws):
            TARGET_VECTOR.append([x for x in NUMPY[row_idx]])

        number_of_labels = len(TARGET_VECTOR[0])  #USING MULTI-OUT, # OUTPUT VECTORS MUST = LEN OF TARGET EXAMPLES

    elif target_method == 'SOFTMAX_POWERBALL_ONLY':

        TARGET_VECTOR.clear()
        NUMPY = n.array(RAW_TARGET_SOURCE['POWERBALL'], dtype=int)
        number_of_draws = vui.validate_user_int('\nEnter number of draws (must be the same as BIG MATRIX) > ', min=1, max=len(NUMPY))

        TARGET_VECTOR = [[] for x in range(n.min(NUMPY),n.max(NUMPY)+1)]
        for row_idx in range(0, len(NUMPY) - number_of_draws):
            for bucket_idx in range(len(TARGET_VECTOR)):
                if bucket_idx + 1 == NUMPY[row_idx]:
                    TARGET_VECTOR[bucket_idx].append(1)
                else:
                    TARGET_VECTOR[bucket_idx].append(0)


        number_of_labels = len(TARGET_VECTOR)  #USING MULTI-OUT, # OUTPUT VECTORS MUST = LEN OF TARGET EXAMPLES


    target_method = 'CUSTOM'

    return target_method, '', '', TARGET_VECTOR, TARGET_VECTOR_HEADER, '', number_of_labels, split_method, '', '', ''

    # return target_method, RAW_TARGET_VECTOR, RAW_TARGET_VECTOR_HEADER, TARGET_VECTOR, TARGET_VECTOR_HEADER, LABEL_RULES,
    #       number_of_labels, split_method, event_value, negative_value, keep_columns




