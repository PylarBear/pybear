from general_list_ops import list_select as ls
from font import font as f
from ML_PACKAGE.GENERIC_PRINT import print_object_create_success as pocs


#ONLY CALLED HERE
def standard_configs_list():
    return [
    'AA',
    'POWERBALL',
    'LEFT1_TOP1',
    'MANUAL ENTRY'
]


#CALLED BY ML
def standard_configs(data_read_manual_config, raw_target_read_manual_config, rv_read_manual_config, data_read_method,
        raw_target_read_method, rv_read_method, BBM_manual_config, base_raw_target_read_manual_config, base_rv_manual_config,
        BBM_build_method, base_raw_target_build_method, base_rv_build_method):

    standard_config = ls.list_single_select(standard_configs_list(),'Select configuration package','value')[0]
    if 'AA' in standard_config:
        data_read_manual_config, raw_target_read_manual_config, rv_read_manual_config = 'N', 'N', 'N'
        data_read_method, raw_target_read_method, rv_read_method = 'STANDARD','STANDARD','STANDARD - ROW_ID & APP_DATE'
        BBM_manual_config, base_raw_target_manual_config, base_rv_manual_config = 'N', 'N', 'N'
        BBM_build_method, base_raw_target_build_method, base_rv_build_method = 'STANDARD', 'STANDARD', 'STANDARD'
    elif 'POWERBALL' in standard_config:
        data_read_manual_config, raw_target_read_manual_config, rv_read_manual_config  = 'N', 'N', 'N'
        data_read_method, raw_target_read_method, rv_read_method = 'STANDARD','STANDARD','NONE - COLUMN OF DUMS'
        BBM_manual_config, base_raw_target_manual_config, base_rv_manual_config = 'N', 'N', 'N'
        BBM_build_method, base_raw_target_build_method, base_rv_build_method = 'STANDARD', 'STANDARD', 'STANDARD'
    elif 'LEFT1_TOP1' in standard_config:
        data_read_manual_config, raw_target_read_manual_config, rv_read_manual_config  = 'N', 'Y', 'N'
        data_read_method, raw_target_read_method, rv_read_method = 'STANDARD','STANDARD','NONE - COLUMN OF DUMS'
        BBM_manual_config, base_raw_target_manual_config, base_rv_manual_config = 'N', 'N', 'N'
        BBM_build_method, base_raw_target_build_method, base_rv_build_method = 'STANDARD', 'STANDARD', 'STANDARD'
    elif 'MANUAL ENTRY' in standard_config:
        data_read_manual_config, raw_target_read_manual_config, rv_read_manual_config = 'Y', 'Y', 'Y'
        BBM_build_method, base_raw_target_build_method, base_rv_build_method = 'STANDARD', 'STANDARD', 'STANDARD'
        BBM_manual_config, base_raw_target_manual_config, base_rv_manual_config = 'N', 'N', 'N'

    return standard_config, data_read_manual_config, raw_target_read_manual_config, rv_read_manual_config, \
            data_read_method, raw_target_read_method, rv_read_method, BBM_manual_config, base_raw_target_manual_config, \
            base_rv_manual_config, BBM_build_method, base_raw_target_build_method, base_rv_build_method

#ONLY CALLED HERE
def top_level_config_print_template(standard_config, object):
    print(f'\nLoading {standard_config.upper()} {object} configs...')


#CALLED BY data_read_config_run
def data_read_standard_configs(standard_config, data_read_method):

    top_level_config_print_template(standard_config, 'FILE READ')

    if standard_config == 'AA':
        from ML_PACKAGE.standard_configs.AA_config import AA_data_read_config as AAdrc
        DATA_OBJECT = AAdrc.AADataReadConfig(standard_config, data_read_method).run()

    elif standard_config == 'POWERBALL':
        from ML_PACKAGE.standard_configs.POWERBALL_config import POWERBALL_data_read_config as pdrc
        DATA_OBJECT = pdrc.PowerballDataReadConfigRun(standard_config, data_read_method).run()

    elif standard_config == 'LEFT1_TOP1':
        from ML_PACKAGE.standard_configs.LEFT1_TOP1_config import LEFT1_TOP1_data_read_config as ldrc
        DATA_OBJECT = ldrc.Left1Top1DataReadConfig(standard_config, data_read_method).run()

    return DATA_OBJECT


#CALLED BY rv_read_config_run
def rv_read_standard_configs(standard_config, rv_read_method, DATA_DF):

    top_level_config_print_template(standard_config, 'RV READ')

    if standard_config == 'AA':
        from ML_PACKAGE.standard_configs.AA_config import AA_rv_read_config as AArrc
        REFERENCE_VECTORS_DF, DATA_DF = AArrc.AARVReadConfig(standard_config, rv_read_method, DATA_DF).run()

    elif standard_config == 'POWERBALL':
        from ML_PACKAGE.standard_configs.POWERBALL_config import POWERBALL_rv_read_config as prrc
        REFERENCE_VECTORS_DF, DATA_DF = prrc.PowerballRVReadConfig(standard_config, rv_read_method, DATA_DF).run()

    elif standard_config == 'LEFT1_TOP1':
        from ML_PACKAGE.standard_configs.LEFT1_TOP1_config import LEFT1_TOP1_rv_read_config as lrrc
        REFERENCE_VECTORS_DF, DATA_DF = lrrc.Left1Top1RVReadConfig(standard_config, rv_read_method, DATA_DF).run()

    return REFERENCE_VECTORS_DF, DATA_DF


#CALLED BY raw_target_read_config_run
def raw_target_read_standard_configs(standard_config, raw_target_read_method, DATA_DF):

    top_level_config_print_template(standard_config, 'RAW TARGET READ')

    if standard_config == 'AA':
        from ML_PACKAGE.standard_configs.AA_config import AA_raw_target_read_config as AArtrc
        RAW_TARGET_DF, DATA_DF = AArtrc.AARawTargetReadConfig(
                    standard_config, raw_target_read_method, DATA_DF).run()

    elif standard_config == 'POWERBALL':
        from ML_PACKAGE.standard_configs.POWERBALL_config import POWERBALL_raw_target_read_config as prtrc
        RAW_TARGET_DF, DATA_DF = prtrc.PowerballRawTargetReadConfig(
                    standard_config, raw_target_read_method, DATA_DF).run()

    elif standard_config == 'LEFT1_TOP1':
        from ML_PACKAGE.standard_configs.LEFT1_TOP1_config import LEFT1_TOP1_raw_target_read_config as lrtrc
        RAW_TARGET_DF, DATA_DF = lrtrc.Left1Top1RawTargetReadConfig(
                    standard_config, raw_target_read_method, DATA_DF).run()

    return RAW_TARGET_DF, DATA_DF


#CALLED BY base_big_matrix_config_run
def BASE_BIG_MATRIX_standard_configs(standard_config, BBM_build_method, NUMPY_OBJECT, NUMPY_HEADER):

    top_level_config_print_template(standard_config, 'BASE BIG MATRIX')

    if standard_config == 'AA':
        from ML_PACKAGE.standard_configs.AA_config import AA_base_big_matrix_config as abbmc
        RAW_DATA_NUMPY_OBJECT, RAW_DATA_NUMPY_HEADER, KEEP = \
            abbmc.AABaseBigMatrixConfig(standard_config, BBM_build_method, NUMPY_OBJECT, NUMPY_HEADER).run()

    elif standard_config == 'POWERBALL':
        from ML_PACKAGE.standard_configs.POWERBALL_config import POWERBALL_BASE_BIG_MATRIX_config as pbbmc
        RAW_DATA_NUMPY_OBJECT, RAW_DATA_NUMPY_HEADER, KEEP = \
            pbbmc.POWERBALLBaseBigMatrixConfig(standard_config, BBM_build_method, NUMPY_OBJECT, NUMPY_HEADER).run()

    elif standard_config == 'LEFT1_TOP1':
        from ML_PACKAGE.standard_configs.LEFT1_TOP1_config import LEFT1_TOP1_BASE_BIG_MATRIX_config as lbbmc
        RAW_DATA_NUMPY_OBJECT, RAW_DATA_NUMPY_HEADER, KEEP = \
            lbbmc.LEFT1TOP1BaseBigMatrixConfig(standard_config, BBM_build_method, NUMPY_OBJECT, NUMPY_HEADER).run()

    return RAW_DATA_NUMPY_OBJECT, RAW_DATA_NUMPY_HEADER, KEEP


#CALLED BY base_raw_target_config_run
def BASE_RAW_TARGET_standard_configs(standard_config, base_raw_target_build_method, NUMPY_OBJECT, NUMPY_HEADER):

    top_level_config_print_template(standard_config, 'BASE RAW TARGET VECTOR')

    if standard_config == 'AA':
        from ML_PACKAGE.standard_configs.AA_config import AA_base_raw_target_config as abrtc
        RAW_TARGET_NUMPY_OBJECT, RAW_TARGET_NUMPY_HEADER, DUM = \
            abrtc.AABaseRawTargetConfig(standard_config, base_raw_target_build_method, NUMPY_OBJECT, NUMPY_HEADER).run()

    elif standard_config == 'POWERBALL':
        from ML_PACKAGE.standard_configs.POWERBALL_config import POWERBALL_base_raw_target_config as pbrtc
        RAW_TARGET_NUMPY_OBJECT, RAW_TARGET_NUMPY_HEADER, DUM = \
            pbrtc.POWERBALLBaseBigMatrixConfig(standard_config, base_raw_target_build_method, NUMPY_OBJECT, NUMPY_HEADER).run()

    elif standard_config == 'LEFT1_TOP1':
        from ML_PACKAGE.standard_configs.LEFT1_TOP1_config import LEFT1_TOP1_base_raw_target_config as lbrtc
        RAW_TARGET_NUMPY_OBJECT, RAW_TARGET_NUMPY_HEADER, DUM = \
            lbrtc.LEFT1TOP1BaseRawTargetConfig(standard_config, base_raw_target_build_method, NUMPY_OBJECT, NUMPY_HEADER).run()

    return RAW_TARGET_NUMPY_OBJECT, RAW_TARGET_NUMPY_HEADER, DUM


#CALLED BY base_reference_vectors_config_run
def BASE_REFERENCE_VECTORS_standard_configs(standard_config, base_rv_build_method, NUMPY_OBJECT, NUMPY_HEADER):

    top_level_config_print_template(standard_config, 'BASE REFERENCE VECTORS')

    if standard_config == 'AA':
        from ML_PACKAGE.standard_configs.AA_config import AA_base_rv_config as abrvc
        RAW_RV_NUMPY_OBJECT, RAW_RV_NUMPY_HEADER, DUM = \
            abrvc.AABaseRVConfig(standard_config, base_rv_build_method, NUMPY_OBJECT, NUMPY_HEADER).run()

    elif standard_config == 'POWERBALL':
        from ML_PACKAGE.standard_configs.POWERBALL_config import POWERBALL_base_rv_config as pbrvc
        RAW_RV_NUMPY_OBJECT, RAW_RV_NUMPY_HEADER, DUM = \
            pbrvc.POWERBALLBaseRVConfig(standard_config, base_rv_build_method, NUMPY_OBJECT, NUMPY_HEADER).run()

    elif standard_config == 'LEFT1_TOP1':
        from ML_PACKAGE.standard_configs.LEFT1_TOP1_config import LEFT1_TOP1_base_rv_config as lbrvc
        RAW_RV_NUMPY_OBJECT, RAW_RV_NUMPY_HEADER, DUM = \
            lbrvc.LEFT1TOP1BaseRVConfig(standard_config, base_rv_build_method, NUMPY_OBJECT, NUMPY_HEADER).run()

    return RAW_RV_NUMPY_OBJECT, RAW_RV_NUMPY_HEADER, DUM


#CALLED BY big_matrix_config_run
def BIG_MATRIX_standard_configs(standard_config, BASE_BIG_MATRIX, BIG_MATRIX, RETAINED_ATTRIBUTES, COLIN_CHOPPED, \
                        MKT, interaction, int_cutoff, intercept, things_in_intercept):

    top_level_config_print_template(standard_config, 'BIG MATRIX')

    if standard_config == 'AA':
        from ML_PACKAGE.standard_configs.AA_config import AA_BIG_MATRIX_config as abmc

        BIG_MATRIX, RETAINED_ATTRIBUTES, TARGET_VECTOR_CHOPPED, APPDATE_VECTOR, COLIN_CHOPPED, \
        interaction, int_cutoff, intercept, things_in_intercept = \
            abmc.AA_big_matrix_config(standard_config, BIG_MATRIX, BASE_BIG_MATRIX, RETAINED_ATTRIBUTES, \
                                      COLIN_CHOPPED, MKT, interaction, int_cutoff, intercept, things_in_intercept)

    elif standard_config == 'POWERBALL':
        from ML_PACKAGE.standard_configs.POWERBALL_config import POWERBALL_BIG_MATRIX_config as pbmc
        # MKT = []
        # METADATA_KEEP = []
        # FILTER_MATRIX = []
        # start_date = ''
        # end_date = ''

        BIG_MATRIX, RETAINED_ATTRIBUTES, TARGET_VECTOR_CHOPPED, interaction, int_cutoff, intercept, \
        things_in_intercept = \
            pbmc.powerball_big_matrix_config(standard_config, BIG_MATRIX, BASE_BIG_MATRIX, RETAINED_ATTRIBUTES,
                            interaction, int_cutoff, intercept, things_in_intercept)

    elif standard_config == 'LEFT1_TOP1':
        from ML_PACKAGE.standard_configs.LEFT1_TOP1_config import LEFT1_TOP1_BIG_MATRIX_config as lbmc

        BIG_MATRIX, RETAINED_ATTRIBUTES, TARGET_VECTOR_CHOPPED, interaction, int_cutoff, intercept, \
        things_in_intercept = \
            lbmc.LEFT1_TOP1_big_matrix_config(standard_config, BIG_MATRIX, BASE_BIG_MATRIX, RETAINED_ATTRIBUTES,
                            interaction, int_cutoff, intercept, things_in_intercept)

    pocs.print_object_create_success(standard_config, 'BIG MATRIX')

    return BIG_MATRIX, TARGET_VECTOR_CHOPPED, RETAINED_ATTRIBUTES, COLIN_CHOPPED, MKT, interaction, \
           int_cutoff, intercept, things_in_intercept







#CALLED BY splitter_config
def TARGET_VECTOR_standard_configs(standard_config, target_config, RAW_TARGET_SOURCE, RAW_TARGET_SOURCE_HEADER, TARGET_VECTOR,
                                   TARGET_VECTOR_HEADER):
    # 1-10-22 THERE ARE 4 POSSIBLE WAYS TO GENERATE TARGET_VECTOR / LABEL_RULES / OTHER RULES (REMEMBER LABEL_RULES IS THE KEY
    # THAT TargetPrerunRun USES TO KNOW WHAT CATEGORIES TO LABEL AS positive/negative)
    # OTHER RULES = split_method, number_of_labels, event_value, negative_value, delete_columns
    # TARGET_VECTOR IS FINAL VECTOR USED IN MATH, SOURCE_TARGET_VECTOR IS TARGET AS READ FROM FILE, RAW_TARGET VECTOR IS
    # AFTER ANY MODIFICATIONS DONE TO SOURCE_TARGET_VECTOR (eg COLUMN CHOPS)
    # 1) TAKE IN SOURCE, CHOP COLUMNS (TURN TO RAW_TARGET) GENERATE LABEL_RULES SELECT OTHER RULES MANUALLY, PROCESS THRU PrerunRun
    # 2) TAKE IN SOURCE, GET ALL LABEL_RULES ETC. FROM standard_config, PROCESS SOURCE AS IS THRU PrerunRun
    # 3) TAKE IN SOURCE, TAKE ALL THE WAY THRU INTO X_standard_config, PROCESS UNIQUELY & RETURN TARGET_VECTOR & HEADER
    # 4) TAKE IN SOURCE, TAKE ALL THE WAY THRU INTO X_standard_config, PROCESS INTO RAW_TARGET, GENERATE LABEL_RULES & OTHER RULES,
    #       BRING ALL OUT TO PrerunConfigRun AND PROCESS THRU PrerunRun


    top_level_config_print_template(standard_config, 'TARGET VECTOR')

    if standard_config == 'AA':
        from ML_PACKAGE.standard_configs.AA_config import AA_target_config as AAtc
        RAW_TARGET_VECTOR, RAW_TARGET_VECTOR_HEADER, TARGET_VECTOR, TARGET_VECTOR_HEADER, LABEL_RULES, \
        number_of_labels, split_method, event_value, negative_value, keep_columns = \
            AAtc.AA_target_configs(standard_config, target_config, RAW_TARGET_SOURCE, RAW_TARGET_SOURCE_HEADER,
                                   TARGET_VECTOR, TARGET_VECTOR_HEADER)

    elif standard_config == 'POWERBALL':
        from ML_PACKAGE.standard_configs.POWERBALL_config import POWERBALL_target_config as ptc
        RAW_TARGET_VECTOR, RAW_TARGET_VECTOR_HEADER, TARGET_VECTOR, TARGET_VECTOR_HEADER, LABEL_RULES, \
        number_of_labels, split_method, event_value, negative_value, keep_columns = \
            ptc.powerball_target_configs(standard_config, target_config, RAW_TARGET_SOURCE, RAW_TARGET_SOURCE_HEADER,
                                         TARGET_VECTOR, TARGET_VECTOR_HEADER)

    elif standard_config == 'LEFT1_TOP1':
        from ML_PACKAGE.standard_configs.LEFT1_TOP1_config import LEFT1_TOP1_target_config as ltc
        RAW_TARGET_VECTOR, RAW_TARGET_VECTOR_HEADER, TARGET_VECTOR, TARGET_VECTOR_HEADER, LABEL_RULES, \
        number_of_labels, split_method, event_value, negative_value, keep_columns = \
            ltc.LEFT1_TOP1_target_configs(standard_config, target_config, RAW_TARGET_SOURCE, RAW_TARGET_SOURCE_HEADER,
                                          TARGET_VECTOR, TARGET_VECTOR_HEADER)

    elif standard_config == 'MANUAL ENTRY':
        # TO PREVENT CRASH IF USER SELECTS A STANDARD CONFIG WHILE IN MANUAL ENTRY
        # BEAR FIX
        print(f'CAPABILITY OF SELECTING A STANDARD TARGET CONFIG WHILE IN MANUAL ENTRY NOT AVAILABLE YET :(')
        RAW_TARGET_VECTOR = []
        RAW_TARGET_VECTOR_HEADER = []
        LABEL_RULES = []
        number_of_labels = ''
        split_method = ''
        event_value = ''
        negative_value = ''
        keep_columns = []


    return RAW_TARGET_VECTOR, RAW_TARGET_VECTOR_HEADER, TARGET_VECTOR, TARGET_VECTOR_HEADER, LABEL_RULES, \
        number_of_labels, split_method, event_value, negative_value, keep_columns


#CALLED BY reference_vectors_config_run
def reference_vectors_standard_config(REFERENCE_VECTORS, standard_config):

    top_level_config_print_template(standard_config, 'REFERENCE VECTORS')

    if standard_config == 'AA':
        from ML_PACKAGE.standard_configs.AA_config import AA_REFERENCE_VECTORS_config as arvc
        REFERENCE_VECTORS = arvc.AA_reference_vectors_config(REFERENCE_VECTORS, standard_config)

    elif standard_config == 'POWERBALL':
        from ML_PACKAGE.standard_configs.POWERBALL_config import POWERBALL_REFERENCE_VECTORS_config as pbrvc
        REFERENCE_VECTORS = pbrvc.powerball_reference_vectors_config(REFERENCE_VECTORS, standard_config)

    elif standard_config == 'LEFT1_TOP1':
        from ML_PACKAGE.standard_configs.LEFT1_TOP1_config import LEFT1_TOP1_REFERENCE_VECTORS_config as lrvc
        REFERENCE_VECTORS = lrvc.LEFT1_TOP1_reference_vectors_config(REFERENCE_VECTORS, standard_config)

    return REFERENCE_VECTORS



#CALLED BY nn_config
# 2-7-22 BEAR NEEDS WORK, PARAMS AND OUTPUTS
def NN_standard_configs(standard_config):

    top_level_config_print_template(standard_config, 'NN')

    if standard_config == 'AA':
        from ML_PACKAGE.standard_configs.AA_config import AA_NN_config as anc
        ARRAY_OF_NODES, NEURONS, nodes, node_seed, activation_constant, aon_base_path, aon_filename, cost_fxn, \
        SELECT_LINK_FXN, LIST_OF_NN_ELEMENTS, gd_iterations, batch_method, gd_method, conv_method, \
        lr_method, LEARNING_RATE, momentum_weight, rglztn_type, rglztn_fctr, conv_kill, pct_change, conv_end_method, \
        non_neg_coeffs, allow_summary_print, summary_print_interval, iteration = anc.AA_NN_config(standard_config)

    elif standard_config == 'POWERBALL':
        from ML_PACKAGE.standard_configs.POWERBALL_config import POWERBALL_NN_config as pnc
        # 2-10-22 BEAR FINISH
        cost_fxn, non_neg_coeffs = pgc.powerball_NN_config(cost_fxn, non_neg_coeffs, standard_config)

    elif standard_config == 'LEFT1_TOP1':
        from ML_PACKAGE.standard_configs.LEFT1_TOP1_config import LEFT1_TOP1_NN_config as lnc
        # 2-10-22 BEAR FINISH
        cost_fxn, non_neg_coeffs = lgc.LEFT1_TOP1_NN_config(cost_fxn, non_neg_coeffs, standard_config)

    elif standard_config == 'MANUAL ENTRY':
        # TO PREVENT CRASH IF USER SELECTS A STANDARD CONFIG WHILE IN MANUAL ENTRY
        # BEAR FIX
        print(f'CAPABILITY OF SELECTING A STANDARD NN CONFIG WHILE IN MANUAL ENTRY NOT AVAILABLE YET :(')
        print(f'USING NN STANDARD CONFIG')
        from ML_PACKAGE.standard_configs.AA_config import AA_NN_config as anc
        ARRAY_OF_NODES, NEURONS, nodes, node_seed, activation_constant, aon_base_path, aon_filename, cost_fxn, \
        SELECT_LINK_FXN, LIST_OF_NN_ELEMENTS, gd_iterations, batch_method, gd_method, conv_method, \
        lr_method, LEARNING_RATE, momentum_weight, rglztn_type, rglztn_fctr, conv_kill, pct_change, conv_end_method, \
        non_neg_coeffs, allow_summary_print, summary_print_interval, iteration = anc.AA_NN_config(standard_config)


    return ARRAY_OF_NODES, NEURONS, nodes, node_seed, activation_constant, aon_base_path, aon_filename, cost_fxn, \
           SELECT_LINK_FXN, LIST_OF_NN_ELEMENTS, gd_iterations, batch_method, gd_method, conv_method, lr_method, \
           LEARNING_RATE, momentum_weight, rglztn_type, rglztn_fctr, conv_kill, pct_change, conv_end_method, \
           non_neg_coeffs, allow_summary_print, summary_print_interval, iteration


#CALLED BY NNRun
def test_cases_calc_standard_configs(standard_config, TEST_MATRIX, BIG_MATRIX, ARRAY_OF_NODES, SELECT_LINK_FXN, \
                                     RETAINED_ATTRIBUTES, activation_constant):

    top_level_config_print_template(standard_config, 'TEST CASES')

    if standard_config == 'AA':
        from ML_PACKAGE.standard_configs.AA_config import AA_test_cases_calc as AAtcc
        CSUTM_DF = AAtcc.AA_test_cases_calc(standard_config, TEST_MATRIX, BIG_MATRIX, ARRAY_OF_NODES, \
                            SELECT_LINK_FXN, RETAINED_ATTRIBUTES, activation_constant)

        return CSUTM_DF

    elif standard_config == 'POWERBALL':
        from ML_PACKAGE.standard_configs.POWERBALL_config import POWERBALL_test_cases_calc as ptcc
        RETAINED_ATTRIBUTES = []
        OUTPUT_VECTOR = ptcc.powerball_test_cases_calc(standard_config, TEST_MATRIX, ARRAY_OF_NODES, \
                                            RETAINED_ATTRIBUTES, SELECT_LINK_FXN, activation_constant)
        return OUTPUT_VECTOR

    elif standard_config == 'LEFT1_TOP1':
        from ML_PACKAGE.standard_configs.LEFT1_TOP1_config import LEFT1_TOP1_test_cases_calc as ltcc
        RETAINED_ATTRIBUTES = []
        OUTPUT_VECTOR = ltcc.LEFT1_TOP1_test_cases_calc(standard_config, TEST_MATRIX, ARRAY_OF_NODES, \
                                            RETAINED_ATTRIBUTES, SELECT_LINK_FXN, activation_constant)
        return OUTPUT_VECTOR

    pocs.print_object_create_success(standard_config, 'TEST CASES')









