import numpy as n, pandas as p
from copy import deepcopy
from data_validation import validate_user_input as vui
from general_list_ops import list_select as ls
from ML_PACKAGE.GENERIC_PRINT import print_post_run_options as ppro
from ML_PACKAGE import MLConfigTemplate as mlct
from ML_PACKAGE.SVM_PACKAGE import SVMCoreConfigCode as sccc
from ML_PACKAGE.SVM_PACKAGE import svm_error_calc as sec
from ML_PACKAGE.standard_configs import standard_configs as sc



# OVERWRITTEN
# standard_config_module()                   return standard config module
# module_specific_config_cmds()              unique config commands for specific ML package
# module_specific_config_str()               allowed menu keystrokes
# module_specific_operations()               parameter selection operations specific to a child module
# print_parameters()                         print parameter state for child module
# return_fxn()                               return from child module

# INHERITED
# config()                                   exe


class SVMConfig(mlct.MLConfigTemplate):

    def __init__(self, standard_config, svm_config, SUPER_WORKING_NUMPY_LIST, WORKING_SUPOBJS, data_run_orientation,
                 margin_type, cost_fxn, kernel_fxn, constant, exponent, sigma, alpha_seed, max_passes, tol, C,
                 alpha_selection_alg, SMO_a2_selection_method, conv_kill, pct_change, conv_end_method, bypass_validation):


        super().__init__(standard_config, svm_config, SUPER_WORKING_NUMPY_LIST, WORKING_SUPOBJS, data_run_orientation,
                         conv_kill, pct_change, conv_end_method, 'dummy rglztn type', C, bypass_validation, __name__)

        # OVERWRITE __init__ ATTRIBUTES
        self.C = self.rglztn_fctr

        # SVM SPECIFIC __init__
        self.DATA = self.SUPER_WORKING_NUMPY_LIST[0]
        self.TARGET = self.SUPER_WORKING_NUMPY_LIST[1]

        self.svm_config = svm_config

        self.margin_type = margin_type
        self.cost_fxn = cost_fxn

        self.kernel_fxn = kernel_fxn
        self.constant = constant
        self.exponent = exponent
        self.sigma = sigma

        self.alpha_seed = alpha_seed
        self.max_passes = max_passes
        self.tol = tol
        self.alpha_selection_alg = alpha_selection_alg
        self.SMO_a2_selection_method = SMO_a2_selection_method

        self.conv_kill = conv_kill
        self.pct_change = pct_change
        self.conv_end_method = conv_end_method

        self.SUPPORT_VECTORS = []
        self.SUPPORT_TARGETS = []
        self.SUPPORT_ALPHAS = []
        self.b = 0

        # SVM SPECIFIC
        self.config_str = ''


    def standard_config_module(self):
        self.margin_type, self.C, self.cost_fxn, self.kernel_fxn, self.constant, self.exponent, self.sigma, \
        self.alpha_seed, self.alpha_selection_alg, self.max_passes, self.tol, self.a2_selection_method = \
            sc.NN_standard_configs(self.standard_config)


    def module_specific_config_cmds(self):            #  unique config commands for specific ML package
        return {
                'b': 'load boundary info from file',
                'c': 'set regularization constant',
                'd': 'set a2 selection method',
                'f': 'change cost function',
                'g': 'select kernel',
                'h': 'alpha selection algorithm',
                'j': 'set margin type',
                'k': 'set max passes',
                'l': 'set alpha seed',
                'm': 'set tolerance',
                'n': 'set polynomial constant',
                'p': 'set polynomial exponent',
                'r': 'set gaussian sigma',
                's': 'adjust convergence kill',
                }


    def module_specific_operations(self):
        ################################################################################################################
        #################################################################################################################

        # DO COMPULSORY CONFIG (IF 'E' WAS CALLED IN THE PARENT)
        if self.sub_config == 'E': self.config_str = 'JCFGLHS'

        else: self.config_str = self.sub_config         # OTHERWISE, JUST RUN THE sub_config AS SELECTED BY USER.

        for sub_config in self.config_str:
            self.SUPPORT_VECTORS, self.SUPPORT_TARGETS, self.SUPPORT_ALPHAS, self.b, self.margin_type, self.C, \
            self.cost_fxn, self.kernel_fxn, self.constant, self.exponent, self.sigma, self.alpha_seed, self.alpha_selection_alg, \
            self.max_passes, self.tol, self.SMO_a2_selection_method, self.conv_kill, self.pct_change, self.conv_end_method = \
            sccc.SVMCoreConfigCode(sub_config, self.margin_type, self.C, self.cost_fxn, self.kernel_fxn, self.constant,
                self.exponent, self.sigma, self.alpha_seed, self.alpha_selection_alg, self.max_passes, self.tol,
                self.SMO_a2_selection_method, self.conv_kill, self.pct_change, self.conv_end_method).config()

        ####################################################################################################################
        ####################################################################################################################


    def print_parameters(self):
        _ = lambda text, width: str(text).ljust(width)
        __ = lambda text, width: str(text).center(width)
        xs = 6
        sh = 12
        lg = 60

        print()

        SVM_PARAMS_PRINT1 = {
            f'margin type': self.margin_type,
            f'C': self.C,
            f'cost fxn': sec.cost_functions()[self.cost_fxn],
            f'kernel fxn': self.kernel_fxn,
            f'polynomial constant': self.constant,
            f'polynomial exponent': self.exponent,
            f'gaussian sigma': self.sigma
        }

        SVM_PARAMS_PRINT2 = {
            f'alpha seed': self.alpha_seed,
            f'alpha selection alg': self.alpha_selection_alg,
            f'max passes': self.max_passes,
            f'tol': self.tol,
            f'SMO_a2_selection_method': self.SMO_a2_selection_method,
            f'passes to convergence kill': self.conv_kill,
            f'conv kill min % change': self.pct_change,
            f'convergence end method': self.conv_end_method
        }


        # CREATE COMBINED DICT FOR SYNCRONIZING PARSING OF PRINTOUTS
        DUM = SVM_PARAMS_PRINT1 | SVM_PARAMS_PRINT2
        max_key_len = n.max(n.fromiter((len(_) for _ in DUM), dtype=int))
        max_value_len = n.max(n.fromiter((len(str(DUM[_])) for _ in DUM), dtype=int))
        del DUM

        print()
        _ = SVM_PARAMS_PRINT1
        __ = SVM_PARAMS_PRINT2
        keys1 = _.keys()
        keys2 = __.keys()
        for line_idx in range(max(len(_), len(__))):
            try: params1_txt = str([*keys1][line_idx]).ljust(max_key_len + 5) + str(_[[*keys1][line_idx]]).ljust(max_value_len)
            except: params1_txt = ' ' * (max_key_len + 5 + max_value_len)

            try: params2_txt = str([*keys2][line_idx]).ljust(max_key_len + 5) + str(__[[*keys2][line_idx]]).ljust(max_value_len)
            except: params2_txt = ' ' * (max_key_len + 5 + max_value_len)

            print(params1_txt + ' ' * 10 + params2_txt)


    def return_fxn(self):

        return self.SUPPORT_VECTORS, self.SUPPORT_TARGETS, self.SUPPORT_ALPHAS, self.b, self.margin_type, self.C, \
               self.cost_fxn, self.kernel_fxn, self.constant, self.exponent, self.sigma, self.alpha_seed, \
               self.alpha_selection_alg, self.max_passes, self.tol, self.SMO_a2_selection_method, self.conv_kill, \
               self.pct_change, self.conv_end_method











if __name__ == '__main__':
    pass


