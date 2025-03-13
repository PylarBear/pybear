import numpy as n, pandas as p
from general_list_ops import list_select as ls
from ML_PACKAGE.standard_configs import config_notification_templates as cnt

# PARENT OF BBMConfigTemplate, BaseRawTargetConfigTemplate, BaseRVReadConfigTemplate, BTMConfigTemplate
class BaseObjectConfigTemplate:
    def __init__(self, standard_config):
        self.standard_config = standard_config

    def load_config_template(self):

        self.method = ls.list_single_select(
            self.build_methods(), f'Select {self.standard_config.upper()} {self.object_name} build method', 'value')[0].upper()

        print(f'\nLoading {self.standard_config.upper()} {object} {method} build config...')

        return self.method

    def config(self):
        if self.method == '':
            return self.load_config_template()
        else:
            return self.method

    def no_config(self):
        print(f'\n{self.standard_config} {self.object_name} build config not set up yet :(')

    def run(self):
        if self.config() == 'NONE':
            self.BASE_OBJECT = self.no_config()

        elif self.config() == 'STANDARD':
            pass

        return self.BASE_OBJECT


# CALLED BY AA_base_big_matrix_config
class BBMConfigTemplate(BaseObjectConfigTemplate):
    def __init__(self, standard_config, BBM_build_method, RAW_DATA_NUMPY_OBJECT, RAW_DATA_NUMPY_HEADER):
        self.method = BBM_build_method
        self.RAW_DATA_NUMPY_OBJECT = RAW_DATA_NUMPY_OBJECT
        self.RAW_DATA_NUMPY_HEADER = RAW_DATA_NUMPY_HEADER
        super().__init__(standard_config)
        self.object_name = 'BBM'

        # HAVE TO def build_methods and def run FOR EACH APPLICATION OF THIS TEMPLATE

        # INHERITED
        # config(self)
        # no_config(self)
        # run(self)


# CALLED BY POWERBALL_raw_target_read_config, LEFT1_TOP1_raw_target_read_config, AA_raw_target_read_config
class BaseRawTargetConfigTemplate(BaseObjectConfigTemplate):
    def __init__(self, standard_config, raw_target_build_method, RAW_TARGET_NUMPY_OBJECT, RAW_TARGET_NUMPY_HEADER):
        self.method = raw_target_build_method
        self.RAW_TARGET_NUMPY_OBJECT = RAW_TARGET_NUMPY_OBJECT
        self.RAW_TARGET_NUMPY_HEADER = RAW_TARGET_NUMPY_HEADER
        super().__init__(standard_config)
        self.object_name = 'BASE RAW TARGET'

        # HAVE TO def build_methods and def run FOR EACH APPLICATION OF THIS TEMPLATE

        # INHERITED
        # config(self)
        # no_config(self)
        # run(self)


# CALLED BY POWERBALL_rv_read_config, LEFT1_TOP1_rv_read_config, AA_rv_read_config
class BaseRVConfigTemplate(BaseObjectConfigTemplate):
    def __init__(self, standard_config, rv_build_method, RAW_RV_NUMPY_OBJECT, RAW_RV_NUMPY_HEADER):
        self.method = rv_build_method
        self.RAW_RV_NUMPY_OBJECT = RAW_RV_NUMPY_OBJECT
        self.RAW_RV_NUMPY_HEADER = RAW_RV_NUMPY_HEADER
        super().__init__(standard_config)
        self.object_name = 'BASE RV'

        # HAVE TO def build_methods and def run FOR EACH APPLICATION OF THIS TEMPLATE

        # INHERITED
        # config(self)
        # no_config(self)
        # run(self)


# CALLED BY POWERBALL_rv_read_config, LEFT1_TOP1_rv_read_config, AA_rv_read_config
class BaseTestMatrixConfigTemplate(BaseObjectConfigTemplate):
    def __init__(self, standard_config, BTM_pre_run_build_method, RAW_TEST_DATA_NUMPY_OBJECT, RAW_TEST_DATA_NUMPY_HEADER):
        self.method = BTM_pre_run_build_method
        self.RAW_TEST_DATA_NUMPY_OBJECT = RAW_TEST_DATA_NUMPY_OBJECT
        self.RAW_TEST_DATA_NUMPY_HEADER = RAW_TEST_DATA_NUMPY_HEADER
        super().__init__(standard_config)
        self.object_name = 'BASE TEST MATRIX'

        # HAVE TO def build_methods and def run FOR EACH APPLICATION OF THIS TEMPLATE

        # INHERITED
        # config(self)
        # no_config(self)
        # run(self)











