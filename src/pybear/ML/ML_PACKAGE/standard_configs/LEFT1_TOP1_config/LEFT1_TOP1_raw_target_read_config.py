import numpy as n, pandas as p
import general_list_ops as ls
from ML_PACKAGE.standard_configs import ReadConfigTemplate as rct


# CALLED BY standard_configs.raw_target_read_standard_configs()
class Left1Top1RawTargetReadConfig(rct.RawTargetReadConfigTemplate):
    def __init__(self, standard_config, raw_target_read_method, DATA_DF):
        super().__init__(standard_config, raw_target_read_method, DATA_DF)
        self.method = raw_target_read_method

    def Left1Top1_raw_target_read_methods(self):
        return [
            'STANDARD',
        ]

    # config(self) INHERITED
    # no_config(self) INHERITED

    def run(self):
        if self.config() == 'STANDARD':
            LEFT_COLUMN = [_ for _ in self.DATA_DF][0]
            self.RAW_TARGET_DF = self.DATA_DF[LEFT_COLUMN]
            self.DATA_DF = self.DATA_DF.drop(columns=LEFT_COLUMN)
            return self.RAW_TARGET_DF, self.DATA_DF

        elif self.config() == 'NONE - COLUMN OF DUMS':
            self.RAW_TARGET_DF = self.no_config()

        return self.RAW_TARGET_DF, self.DATA_DF