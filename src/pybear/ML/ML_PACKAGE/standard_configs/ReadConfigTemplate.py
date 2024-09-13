import numpy as n, pandas as p
from general_list_ops import list_select as ls
from ML_PACKAGE.standard_configs import config_notification_templates as cnt

# PARENT OF DataReadConfigTemplate, RawTargetReadConfigTemplate, RVReadConfigTemplate, TestDataReadConfigTemplate
class ReadConfigTemplate:
    def __init__(self, standard_config, method, DATA_DF):
        self.standard_config = standard_config
        self.method = method
        self.DATA_DF = DATA_DF
        self.object_name = 'DUM'

    def read_methods(self):
        return [
            'STANDARD'
        ]

    def load_config_template(self):

        method = ls.list_single_select(
            self.read_methods(), f'Select {self.standard_config.upper()} {self.object_name} method', 'value')[0].upper()

        print(f'\nLoading {self.standard_config.upper()} {object} {method} config...')

        return method

    def config(self):
        if self.method == '':
            return self.load_config_template()
        else:
            return self.method

    def no_config(self):
        print(f'\n{self.standard_config} {self.object_name} read config not set up yet :( just making a column of DUMS')

        return p.DataFrame(data=['DUM' for idx in range(len(self.DATA_DF))], columns=['DUM'])

    def run(self):
        if self.config() == 'NONE':
            self.OBJECT_DF = self.no_config()

        elif self.config() == 'STANDARD':
            pass

        return self.OBJECT_DF, self.DATA_DF


# CALLED BY POWERBALL_data_read_config, LEFT1_TOP1_data_read_config, AA_data_read_config
class DataReadConfigTemplate(ReadConfigTemplate):
    def __init__(self, standard_config, data_read_method):
        super().__init__(standard_config, data_read_method, '')
        self.standard_config = standard_config
        self.method = data_read_method
        self.object_name = 'BBM'

        # read_methods(self) inherited
        # config(self) inherited
        # no_config(self) inherited
        # run(self) inherited


# CALLED BY POWERBALL_raw_target_read_config, LEFT1_TOP1_raw_target_read_config, AA_raw_target_read_config
class RawTargetReadConfigTemplate(ReadConfigTemplate):
    def __init__(self, standard_config, raw_target_read_method, DATA_DF):
        super().__init__(standard_config, raw_target_read_method, DATA_DF)
        self.standard_config = standard_config
        self.method = raw_target_read_method
        self.DATA_DF = DATA_DF
        self.object_name = 'RAW TARGET'

        # read_methods(self) inherited
        # config(self) inherited
        # no_config(self) inherited
        # run(self) inherited


# CALLED BY POWERBALL_rv_read_config, LEFT1_TOP1_rv_read_config, AA_rv_read_config
class RVReadConfigTemplate(ReadConfigTemplate):
    def __init__(self, standard_config, rv_read_method, DATA_DF):
        super().__init__(standard_config, rv_read_method, DATA_DF)
        self.standard_config = standard_config
        self.method = rv_read_method
        self.DATA_DF = DATA_DF
        self.object_name = 'RV'

        # read_methods(self) inherited
        # config(self) inherited
        # no_config(self) inherited
        # run(self) inherited


# CALLED BY POWERBALL_test_data_read_config, LEFT1_TOP1_test_data_read_config, AA_test_data_read_config
class TestDataReadConfigTemplate(ReadConfigTemplate):
    def __init__(self, standard_config, rv_read_method, DATA_DF):
        super().__init__(standard_config, rv_read_method, DATA_DF)
        self.standard_config = standard_config
        self.method = rv_read_method
        self.DATA_DF = DATA_DF
        self.object_name = 'TEST DATA'

        # read_methods(self) inherited
        # config(self) inherited
        # no_config(self) inherited
        # run(self) inherited












