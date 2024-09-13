import numpy as np
from ML_PACKAGE import MLConfigTemplate as mlct
from ML_PACKAGE.MLREGRESSION import MLRegressionCoreConfigCode as mlrccc
from MLObjects.SupportObjects import master_support_object_dict as msod


# OVERWRITTEN
# module_specific_config_cmds()              unique config commands for specific ML package
# standard_config_module()                   return standard config module
# module_specific_operations()               parameter selection operations specific to a child module
# print_parameters()                         print parameter state for child module
# return_fxn()                               return from child module

# INHERITED
# config()                                   exe


class MLRegressionConfig(mlct.MLConfigTemplate):
    def __init__(self, standard_config, mlr_config, SUPER_WORKING_NUMPY_LIST, WORKING_SUPOBJS, data_run_orientation,
                 rglztn_type, rglztn_fctr, batch_method, batch_size, intcpt_col_idx, bypass_validation):

        # NOTES 2-6-22 --- REMEMBER WE ARE TAKING IN mlr_config, BUT IT'S BEING __init__ED AS sub_config in
        # THE PARENT (PARENT ONLY TALKS IN sub_config) SO MUST CALL mlr_config AS sub_config HERE

        super().__init__(standard_config, mlr_config, SUPER_WORKING_NUMPY_LIST, WORKING_SUPOBJS, data_run_orientation,
                         'dum_conv_kill', 'dum_pct_change', 'dum_conv_end_method', rglztn_type, rglztn_fctr, bypass_validation,
                         __name__)


        self.batch_method = batch_method
        self.batch_size = batch_size
        self.intcpt_col_idx = intcpt_col_idx


    def module_specific_config_cmds(self):
        return {
                'b': 'regularization type',
                'c': 'regularization factor',
                'd': 'batch method',
                'f': 'batch size'
                }           # SPECIFIED IN CHILDREN, CANT USE AEIOQUZ



    def standard_config_module(self):
        #return sc.MLR_standard_configs(self.standard_config, self.su_config)
        print(f'\n*** STANDARD MLR SETUP CONFIGS NOT AVAILABLE YET :( ***\n')
        pass


    def module_specific_operations(self):
        self.rglztn_type, self.rglztn_fctr, self.batch_method, self.batch_size = \
            mlrccc.MLRegressionCoreConfigCode(self.sub_config, self.SUPER_WORKING_NUMPY_LIST[0],
                self.WORKING_SUPOBJS[0][msod.QUICK_POSN_DICT()["HEADER"]].reshape((1,-1)), self.data_run_orientation,
                self.rglztn_type, self.rglztn_fctr, self.batch_method, self.batch_size).config()

        # TO UPDATE FIELDS IN DYNAMIC MENU
        self.ALL_CMDS = self.GENERIC_CONFIG_CMDS | self.module_specific_config_cmds()


    def print_parameters(self):
        _width = 30
        print(f"RGLZTN TYPE = ".ljust(_width) + f"{self.rglztn_type}")
        print(f"RGLZTN FACTOR = ".ljust(_width) + f"{self.rglztn_fctr}")
        print(f"BATCH METHOD = ".ljust(_width) + f"{dict({'B':'BATCH','M':'MINI-BATCH'})[self.batch_method]}")
        print(f"BATCH SIZE = ".ljust(_width) + f"{self.batch_size}")
        print(f"INTERCEPT COLUMN INDEX = ".ljust(_width) + f"{self.intcpt_col_idx}")


    def return_fxn(self):
        return *self.return_fxn_base(), self.rglztn_type, self.rglztn_fctr, self.batch_method, self.batch_size, \
            self.intcpt_col_idx









if __name__ == '__main__':
    pass






















