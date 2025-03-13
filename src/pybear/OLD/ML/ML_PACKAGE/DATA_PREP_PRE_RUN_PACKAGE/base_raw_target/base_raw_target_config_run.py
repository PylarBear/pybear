import numpy as n, pandas as p
from ML_PACKAGE.DATA_PREP_PRE_RUN_PACKAGE.base_big_matrix import base_big_matrix_config_run as bbmcr
from font import font as f




# CALLED BY class BaseRawTargetBuild (INSIDE filename=class_BaseBMRVTVTMBuild)
class BaseRawTargetConfigRun(bbmcr.BBMConfigRun):
    def __init__(self, standard_config, user_manual_or_standard, base_raw_target_build_method, object_name,
                 RAW_TARGET_NUMPY_OBJECT, RAW_TARGET_NUMPY_HEADER):
        self.standard_config = standard_config
        self.user_manual_or_std = user_manual_or_standard
        self.method = base_raw_target_build_method
        self.object_name = object_name
        self.NUMPY_OBJECT = RAW_TARGET_NUMPY_OBJECT
        self.NUMPY_HEADER = RAW_TARGET_NUMPY_HEADER
        # super().__init__(standard_config, user_manual_or_standard, BBM_build_method, object_name,
        #                                               RAW_DATA_NUMPY_OBJECT, RAW_DATA_NUMPY_HEADER)

    # INHERITED
    # print_results(self)
    # config_run(self)


    def standard_config_source(self):
        from ML_PACKAGE.standard_configs import standard_configs as sc
        return sc.BASE_RAW_TARGET_standard_configs(self.standard_config, self.method,
                                                   self.NUMPY_OBJECT, self.NUMPY_HEADER)


    def print_results(self):
        _ = self.NUMPY_OBJECT
        __ = self.NUMPY_HEADER

        print(f'\n{self.object_name} pre-filter stats:')
        print(
            f'{len(_)} columns and {len(_[0])} rows not counting header')

        display_columns = 10
        print(f'\nFinal NUMPY {self.object_name} as dataframe[:{display_columns}][:20] for display only:')
        TEST_DF = p.DataFrame(data=_.transpose(), columns=__[0].transpose())
        print(TEST_DF[[_ for _ in TEST_DF][:display_columns]].head(20))


    def menu_options(self):
        return ['accept config / continue(a)',
                'bypass(b)',
                'placeholder(f)',
                'placeholder(p)',
                'placeholder(s)',
                'reconfigure all(z)'
                ]






