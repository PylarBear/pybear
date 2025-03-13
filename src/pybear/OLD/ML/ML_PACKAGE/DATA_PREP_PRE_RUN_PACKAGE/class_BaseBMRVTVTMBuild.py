from data_validation import validate_user_input as vui
from ML_PACKAGE.DATA_PREP_PRE_RUN_PACKAGE.base_big_matrix import base_big_matrix_config_run as bbmcr
from ML_PACKAGE.DATA_PREP_PRE_RUN_PACKAGE.base_raw_target import base_raw_target_config_run as btvcr
from ML_PACKAGE.DATA_PREP_PRE_RUN_PACKAGE.base_reference_vector import base_reference_vector_config_run as brvcr


# TEMPLATE FOR BBMBuild, BaseRawTargetBuild, BaseRVBuild, BTMBuild
class BaseBMRVTVTMBuild:
    # BRING IN ALL RAW OBJECTS IN CASE BUILD OF ONE EVER REQUIRES ROW CHOPPING, THEN ALL NEED CHOPS
    def __init__(self, standard_config, object_name, SUPER_RAW_NUMPY_TUPLE):
        self.standard_config = standard_config
        self.object_name = object_name
        print(f'\nPERFORM DATA-SPECIFIC CONDITIONING TO {self.object_name}?\n')
        self.user_manual_or_std = self.user_select_config()
        self.SUPER_RAW_NUMPY_TUPLE = SUPER_RAW_NUMPY_TUPLE
        self.SUPER_RAW_NUMPY_LIST = list(self.SUPER_RAW_NUMPY_TUPLE)  # CONVERT TO LIST SO CAN BE MODIFIED
        # FOR REFERENCE
        # self.RAW_DATA_NUMPY_OBJECT = SUPER_RAW_NUMPY_TUPLE[0]
        # self.RAW_DATA_NUMPY_HEADER = SUPER_RAW_NUMPY_TUPLE[1]
        # self.RAW_TARGET_NUMPY_OBJECT = SUPER_RAW_NUMPY_TUPLE[2]
        # self.RAW_TARGET_NUMPY_HEADER = SUPER_RAW_NUMPY_TUPLE[3]
        # self.RAW_RV_NUMPY_OBJECT = SUPER_RAW_NUMPY_TUPLE[4]
        # self.RAW_RV_NUMPY_HEADER = SUPER_RAW_NUMPY_TUPLE[5]


    # PROMPT USER TO CHOOSE STANDARD CONFIG OR OVERRIDE
    def user_select_config(self):
        # self.manual_config NOT DEFINED HERE IN PARENT, DEFINED IN EACH CHILD
        if self.standard_config == 'MANUAL ENTRY' or self.manual_config == 'Y':

            while True:
                self.user_manual_or_std = vui.validate_user_str('Select a standard config(s), bypass(b) or manual entry(z) > ', 'SBZ')
                if self.user_manual_or_std == 'Z':
                    print(f'MANUAL BUILD OF {self.object_name} NOT AVAILABLE')
                else:
                    break

        elif self.manual_config == 'N':
            self.user_manual_or_std = 'S'

        return self.user_manual_or_std


    def object_idx(self):
        pass # BUILT IN CHILDREN


    def header_idx(self):
        pass # BUILT IN CHILDREN


    def config_build_source_params(self):
        return tuple(
                    [self.standard_config,
                    self.user_manual_or_std,
                    self.method,
                    self.object_name,
                    self.SUPER_RAW_NUMPY_LIST[self.object_idx],
                    self.SUPER_RAW_NUMPY_LIST[self.header_idx]
                      ]
                )


    def build(self):

        # try:
        self.SUPER_RAW_NUMPY_LIST[self.object_idx], self.SUPER_RAW_NUMPY_LIST[self.header_idx], KEEP = \
            self.config_build_source()
        # config_build_source NOT AVAILABLE IN TEMPLATE, BUILT IN EACH OF THE SUBCLASSES

        # except:
        #     eip.exc_info_print('execute', f'{self.object_name} build')
        #     if vui.validate_user_str('Try again(b), or quit(q) > ', 'BQ') == 'B':
        #         pass # 12-1-2021 THIS WAS "CONTINUE" (BUT NOT IN A LOOP) CHANGED TO "PASS" W/O DUE DILIGENCE
        #     else:
        #         sys.exit(f'User terminated.')

        return self.SUPER_RAW_NUMPY_LIST, KEEP



class BBMBuild(BaseBMRVTVTMBuild):
    def __init__(self, standard_config, BBM_manual_config, BBM_build_method, SUPER_RAW_NUMPY_TUPLE,
                 object_name='BASE BIG MATRIX'):
        self.manual_config = BBM_manual_config
        self.method = BBM_build_method
        super().__init__(standard_config, object_name, SUPER_RAW_NUMPY_TUPLE)
        self.object_idx = 0
        self.header_idx = 1


    def config_build_source(self):
        return bbmcr.BBMConfigRun(*self.config_build_source_params()).config_run()
    # INHERITED
    # self.user_select_config()
    # self.build()


class BaseRawTargetBuild(BaseBMRVTVTMBuild):
    def __init__(self, standard_config, base_raw_target_manual_config, base_raw_target_build_method,
                 SUPER_RAW_NUMPY_TUPLE, object_name='RAW TARGET SOURCE'):
        self.manual_config = base_raw_target_manual_config
        self.method = base_raw_target_build_method
        super().__init__(standard_config, object_name, SUPER_RAW_NUMPY_TUPLE)
        self.object_idx = 2
        self.header_idx = 3


    def config_build_source(self):
        return btvcr.BaseRawTargetConfigRun(*self.config_build_source_params()).config_run()

    # INHERITED
    # self.user_select_config()
    # self.build()


class BaseRVBuild(BaseBMRVTVTMBuild):
    def __init__(self, standard_config, base_rv_manual_config, base_rv_build_method, SUPER_RAW_NUMPY_TUPLE,
                 object_name='REFERENCE VECTORS'):
        self.manual_config = base_rv_manual_config
        self.method = base_rv_build_method
        super().__init__(standard_config, object_name, SUPER_RAW_NUMPY_TUPLE)
        self.object_idx = 4
        self.header_idx = 5


    def config_build_source(self):
        return brvcr.BaseRVConfigRun(*self.config_build_source_params()).config_run()

    # INHERITED
    # self.user_select_config()
    # self.build()











