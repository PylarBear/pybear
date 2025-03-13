import numpy as n, pandas as p
import general_list_ops as ls
from ML_PACKAGE.standard_configs import ReadConfigTemplate as rct


# CALLED BY standard_configs.rv_read_standard_configs()
class AARVReadConfig(rct.RVReadConfigTemplate):
    def __init__(self, standard_config, rv_read_method, DATA_DF):
        super().__init__(standard_config, rv_read_method, DATA_DF)
        # self.standard_config = standard_config
        # self.method = rv_read_method
        # self.DATA_DF = DATA_DF

    def read_methods(self):
        return [
            'STANDARD - ROW_ID & APP_DATE'
        ]

    # config.(self) INHERITED
    # no_config(self) INHERITED

    def run(self):
        if self.config() == 'NONE':
            self.OBJECT_DF = self.no_config()

        elif self.config() == 'STANDARD - ROW_ID & APP_DATE':
            self.REFERENCE_VECTORS_DF = self.DATA_DF[['ROW_ID', 'APP DATE']]
            # DONT DROP APP DATE, KEEP IT TO PUT M D Y WEEKDAY IN BBM, THEN DELETE IT
            self.DATA_DF = self.DATA_DF.drop(columns=['ROW_ID'])

        return self.REFERENCE_VECTORS_DF, self.DATA_DF








