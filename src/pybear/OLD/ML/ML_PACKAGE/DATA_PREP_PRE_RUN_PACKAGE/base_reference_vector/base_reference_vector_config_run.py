from ML_PACKAGE.DATA_PREP_PRE_RUN_PACKAGE.base_big_matrix import base_big_matrix_config_run as bbmcr


# CALLED BY class BaseRVBuild (INSIDE filename=class_BaseBMRVTVBuild)
class BaseRVConfigRun(bbmcr.BBMConfigRun):
    def __init__(self, standard_config, user_manual_or_standard, base_rv_build_method, object_name,
                 RAW_RV_NUMPY_OBJECT, RAW_RV_NUMPY_HEADER):
        self.standard_config = standard_config
        self.user_manual_or_std = user_manual_or_standard
        self.method = base_rv_build_method
        self.object_name = object_name
        self.NUMPY_OBJECT = RAW_RV_NUMPY_OBJECT
        self.NUMPY_HEADER = RAW_RV_NUMPY_HEADER
        # super().__init__(standard_config, user_manual_or_standard, BBM_build_method, object_name,
        #                  RAW_DATA_NUMPY_OBJECT, RAW_DATA_NUMPY_HEADERe)

    # INHERITED
    # print_results(self):
    # config_run(self):


    def standard_config_source(self):
        from ML_PACKAGE.standard_configs import standard_configs as sc
        return sc.BASE_REFERENCE_VECTORS_standard_configs(self.standard_config, self.method,
                                        self.NUMPY_OBJECT, self.NUMPY_HEADER)


    def menu_options(self):
        return ['accept config / continue(a)',
                'bypass(b)',
                'placeholder(f)',
                'placeholder(p)',
                'placeholder(s)',
                'reconfigure all(z)'
                ]






























