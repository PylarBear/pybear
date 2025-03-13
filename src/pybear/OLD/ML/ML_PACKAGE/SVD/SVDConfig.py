from ML_PACKAGE import MLConfigTemplate as mlct
from ML_PACKAGE.SVD import SVDCoreConfigCode as sccc


# OVERWRITTEN
# module_specific_config_cmds()              unique config commands for specific ML package
# module_specific_config_str()               allowed menu keystrokes
# standard_config_module()                   return standard config module
# module_specific_operations()               parameter selection operations specific to a child module
# print_parameters()                         print parameter state for child module
# return_fxn()                               return from child module

# INHERITED
# config()                                   exe


class SVDConfig(mlct.MLConfigTemplate):
    def __init__(self, standard_config, svd_config, DATA, DATA_HEADER, rglztn_fctr, conv_kill, pct_change, conv_end_method,
                 svd_max_columns):

        # MODULE SPECIFIC __init__
        self.DATA = DATA
        self.DATA_HEADER = DATA_HEADER
        self.max_columns = svd_max_columns


        # NOTES 4-17-22 --- REMEMBER WE ARE TAKING IN svd_conig, BUT IT'S BEING __init__ED AS sub_config in
        # THE PARENT (PARENT ONLY TALKS IN sub_config) SO MUST INVOKE svd_config THRU sub_config HERE

        super().__init__(standard_config, svd_config, conv_kill, pct_change, conv_end_method, rglztn_fctr, __name__)


    def module_specific_config_cmds(self):
        return {
            'b': 'max columns'
            }           # SPECIFIED IN CHILDREN


    def standard_config_module(self):
        #return sc.SVD_standard_configs(self.standard_config, self.su_config)
        pass


    def module_specific_operations(self):

        self.max_columns = \
            sccc.SVDCoreConfigCode(self.sub_config, self.DATA, self.DATA_HEADER, self.max_columns).config()


    def print_parameters(self):
        print()
        print(f'MAX COLUMNS = '.ljust(30) + f'{self.max_columns}')


    def return_fxn(self):
        return self. max_columns








if __name__ == '__main__':
    standard_config = 'BYPASS'
    svd_config = 'BYPASS'
    DATA = [
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6]
    ]
    DATA_HEADER = [['A','B','C','D']]
    rglztn_fctr = 0.5
    conv_kill = 1000
    pct_change = 0.1
    conv_end_method = 'KILL'
    svd_max_columns = 10

    max_columns = \
    SVDConfig(standard_config, svd_config, DATA, DATA_HEADER, rglztn_fctr, conv_kill, pct_change, conv_end_method,
    svd_max_columns).config()






















