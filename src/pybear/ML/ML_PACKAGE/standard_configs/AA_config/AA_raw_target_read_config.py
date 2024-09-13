import numpy as n, pandas as p
import general_list_ops as ls
from ML_PACKAGE.standard_configs import ReadConfigTemplate as rct



class AARawTargetReadConfig(rct.RawTargetReadConfigTemplate):
    def __init__(self, standard_config, raw_target_read_method, DATA_DF):
        super().__init__(standard_config, raw_target_read_method, DATA_DF)
        self.method = raw_target_read_method

    def AA_raw_target_read_methods(self):
        return [
            'STANDARD'
        ]

    # config(self) INHERITED
    # no_config(self) INHERITED

    def run(self):
        self.RAW_TARGET_VECTOR = p.DataFrame(data=self.DATA_DF['STATUS'], columns=['STATUS'])
        self.DATA_DF = self.DATA_DF.drop(columns=['STATUS'])
        return self.RAW_TARGET_VECTOR, self.DATA_DF
