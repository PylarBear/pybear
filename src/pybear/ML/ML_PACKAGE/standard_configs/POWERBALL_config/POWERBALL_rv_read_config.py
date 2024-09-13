import numpy as n, pandas as p
import general_list_ops as ls
from ML_PACKAGE.standard_configs import ReadConfigTemplate as rct

# CALLED BY standard_configs.rv_read_standard_configs()
class PowerballRVReadConfig(rct.RVReadConfigTemplate):
    def __init__(self, standard_config, rv_read_method, DATA_DF):
        super().__init__(standard_config, rv_read_method, DATA_DF)

    def powerball_RV_read_methods(self):
        return [
            'NONE - COLUMN OF DUMS'
        ]

    # config(self) INHERITED
    # no_config(self) INHERITED

    def run(self):
        if self.config() == 'STANDARD':
            pass

        elif self.config() == 'NONE - COLUMN OF DUMS':
            self.REFERENCE_VECTORS_DF = self.no_config()

        return self.REFERENCE_VECTORS_DF, self.DATA_DF




