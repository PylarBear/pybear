import numpy as n, pandas as p
import general_list_ops as ls
from ML_PACKAGE.standard_configs import ReadConfigTemplate as rct


# CALLED BY standard_configs.raw_target_read_standard_configs()
class PowerballRawTargetReadConfig(rct.RawTargetReadConfigTemplate):
    def __init__(self, standard_config, raw_target_read_method, DATA_DF):
        super().__init__(standard_config, raw_target_read_method, DATA_DF)
        self.method = raw_target_read_method

    def POWERBALL_raw_target_read_methods(self):
        return [
            'STANDARD',
            'NONE'
        ]

    # config(self) INHERITED
    # no_config(self) INHERITED

    def run(self):
        if self.config() == 'STANDARD':
            self.RAW_TARGET_DF = self.DATA_DF
            return self.RAW_TARGET_DF, self.DATA_DF

        elif self.config() == 'NONE - COLUMN OF DUMS':
            self.RAW_TARGET_DF = self.no_config()

        return self.RAW_TARGET_DF, self.DATA_DF




