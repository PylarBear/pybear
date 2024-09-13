from ML_PACKAGE.standard_configs import config_notification_templates as cnt

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

# CALLED ONLY HERE
def LEFT1_TOP1_target_methods():
    return ['METHODS_MSG() LEFT1_TOP1 TARGET CONFIGS NOT AVAILABLE YET :('
            ]


#CALLED BY standard_configs.standard_configs.TARGET_VECTOR_standard_configs()
def LEFT1_TOP1_target_configs(standard_config, target_config, RAW_TARGET_SOURCE, RAW_TARGET_HEADER_SOURCE, TARGET_VECTOR,
                              TARGET_VECTOR_HEADER):

    target_method = cnt.load_config_template(standard_config, 'TARGET VECTOR', LEFT1_TOP1_target_methods())

    if target_method == 'SOFTMAX':
        pass

    elif target_method == 'SVM':
        pass

    elif target_method == 'BINARY':
        pass

    elif target_method == '(METHODS_MSG) LEFT1_TOP1 TARGET CONFIGS NOT AVAILABLE YET :(':
        print(f'(TARGET_CONFIG() MSG) LEFT1_TOP1 TARGET CONFIGS NOT AVAILABLE YET :(')

    if target_method in ['SOFTMAX','SVM','BINARY']:
        return RAW_TARGET_VECTOR, RAW_TARGET_VECTOR_HEADER, TARGET_VECTOR, TARGET_VECTOR_HEADER, LABEL_RULES, \
               number_of_labels, split_method, event_value, negative_value, keep_columns





