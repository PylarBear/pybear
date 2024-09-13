import sys, inspect
import numpy as np, pandas as pd
from copy import deepcopy
from debug import get_module_name as gmn
from data_validation import validate_user_input as vui, arg_kwarg_validater as akv
from general_data_ops import get_shape as gs
from general_list_ops import list_select as ls
from ML_PACKAGE.GENERIC_PRINT import print_post_run_options as ppro
from ML_PACKAGE import MLConfigTemplate as mlct
from ML_PACKAGE.NN_PACKAGE import NNCoreConfigCode as nccc
from ML_PACKAGE.NN_PACKAGE.gd_run import error_calc as ec
from ML_PACKAGE.standard_configs import standard_configs as sc



# OVERWRITTEN
# standard_config_module()                   return standard config module
# module_specific_config_cmds()              unique config commands for specific ML package
# module_specific_operations()               parameter selection operations specific to a child module
# print_parameters()                         print parameter state for child module
# return_fxn()                               return from child module

# INHERITED
# config()                                   exe


class NNConfig(mlct.MLConfigTemplate):
    def __init__(self, standard_config, nn_config, SUPER_WORKING_NUMPY_LIST, WORKING_SUPOBJS, data_run_orientation,
                 target_run_orientation, ARRAY_OF_NODES, NEURONS, nodes, node_seed, activation_constant, aon_base_path,
                 aon_filename, cost_fxn, SELECT_LINK_FXN, LIST_OF_NN_ELEMENTS, OUTPUT_VECTOR, batch_method, BATCH_SIZE,
                 gd_method, conv_method, lr_method, LEARNING_RATE, momentum_weight, rglztn_type, rglztn_fctr, conv_kill,
                 pct_change, conv_end_method, gd_iterations, non_neg_coeffs, allow_summary_print, summary_print_interval,
                 iteration, bypass_validation):

        super().__init__(standard_config, nn_config, SUPER_WORKING_NUMPY_LIST, WORKING_SUPOBJS, data_run_orientation,
                         conv_kill, pct_change, conv_end_method, rglztn_type, rglztn_fctr, bypass_validation, __name__)

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = inspect.stack()[0][3]


        self.DATA = SUPER_WORKING_NUMPY_LIST[0]
        self.TARGET_VECTOR = SUPER_WORKING_NUMPY_LIST[1]
        self.data_run_orientation = akv.arg_kwarg_validater(data_run_orientation, 'data_run_orientation', ['ROW', 'COLUMN'], self.this_module, fxn)
        self.target_run_orientation = akv.arg_kwarg_validater(target_run_orientation, 'target_run_orientation', ['ROW', 'COLUMN'], self.this_module, fxn)
        self.ARRAY_OF_NODES = ARRAY_OF_NODES
        self.NEURONS = NEURONS
        self.nodes = nodes
        self.node_seed = node_seed
        self.activation_constant = activation_constant
        self.aon_base_path = aon_base_path
        self.aon_filename = aon_filename
        self.cost_fxn = cost_fxn.upper()
        self.SELECT_LINK_FXN = SELECT_LINK_FXN
        self.LIST_OF_NN_ELEMENTS = LIST_OF_NN_ELEMENTS
        self.OUTPUT_VECTOR = OUTPUT_VECTOR
        self.batch_method = batch_method
        self.BATCH_SIZE = BATCH_SIZE
        self.gd_method = gd_method
        self.conv_method = conv_method
        self.lr_method = lr_method
        self.LEARNING_RATE = LEARNING_RATE
        self.momentum_weight = momentum_weight
        self.gd_iterations = gd_iterations
        self.non_neg_coeffs = non_neg_coeffs
        self.allow_summary_print = allow_summary_print
        self.summary_print_interval = summary_print_interval
        self.iteration = iteration

        # NN SPECIFIC
        self.config_str = ''

        self.data_rows, self.data_cols = gs.get_shape('DATA', self.DATA, self.data_run_orientation)


    def standard_config_module(self):
        self.ARRAY_OF_NODES, self.NEURONS, self.nodes, self.node_seed, self.activation_constant, self.aon_base_path, \
        self.aon_filename, self.cost_fxn, self.SELECT_LINK_FXN, self.LIST_OF_NN_ELEMENTS, self.gd_iterations, \
        self.batch_method, self.gd_method, self.conv_method, self.lr_method, self.LEARNING_RATE, self.momentum_weight, \
        self.rglztn_type, self.rglztn_fctr, self.conv_kill, self.pct_change, self.conv_end_method, self.non_neg_coeffs, \
        self.allow_summary_print, self.summary_print_interval, self.iteration = sc.NN_standard_configs(self.standard_config)


    def module_specific_config_cmds(self):            #  unique config commands for specific ML package
        return {
                'b': 'batch_method',
                'c': 'convergence method',
                'd': 'activation constant',
                'f': 'cost function',
                'g': 'descent method',
                'h': 'learning rate method',
                'j': 'learning rates',
                'k': 'convergence kill',
                'l': 'link functions',
                'm': 'momentum weight',
                'n': 'non-negative NN parameters',
                'p': 'reset node seed',
                'r': 'initialize nodes',
                's': 'regularization type/factor',
                't': 'allow summary print / interval',
                'v': 'iterations',
                'w': 'restart iteration counter'
                }

        # CAN USE                     TUV    0123456789
        # CANT USE A   E   I       QR       Z
        # USED      BCD FGH JKLMNOP  S   WXY


    def module_specific_operations(self):
        ################################################################################################################
        #################################################################################################################
        # IF USER WANTS TO PROCEED, CHECK IF ARRAY_OF_NODES, BATCH_SIZE, & LEARNING_RATE ARE SIZED CORRECTLY B4 ALLOWING EXIT
        if self.sub_config == 'A':
            _ = False
            if len(self.ARRAY_OF_NODES) != self.nodes: _ = True

            for node_idx in range(len(self.ARRAY_OF_NODES)):
                if len(self.ARRAY_OF_NODES[node_idx]) != self.NEURONS[node_idx]: _ = True; break

                if node_idx == 0 and self.ARRAY_OF_NODES[node_idx].shape[1] != self.data_cols: _ = True; break
                elif node_idx > 0 and \
                    self.ARRAY_OF_NODES[node_idx].shape[1] != self.NEURONS[node_idx-1] + self.activation_constant: _ = True; break

            __ = False
            if len(self.BATCH_SIZE) != self.gd_iterations:
                __ = True

            ___ = False
            if len(self.LEARNING_RATE) != len(self.ARRAY_OF_NODES) or \
                False in map(lambda X: len(X) == self.gd_iterations, self.LEARNING_RATE):
                ___ = True

            ____ = False
            if not (np.min(self.TARGET_VECTOR)==0 and np.max(self.TARGET_VECTOR)==1) and \
                    (self.cost_fxn != 'S' or self.SELECT_LINK_FXN[-1].upper() != 'NONE'):
                ____ = True

            if _:
                print(f'\n*** ARRAY OF NODES IS NOT PROPERLY SIZED.  MUST INITIALIZE NODES TO BUILD ARRAY OF NODES WITH CURRENT')
                print(f'HYPERPARAMETERS OR INVOKE A STANDARD CONFIG TO LOAD HYPERPARAMETERS AND FORCE BUILD OF ARRAY OF NODES ***\n')
                self.sub_config = 'BYPASS'  # CHANGE FROM 'A', FORCE BYPASS OF 'A' IN MLConfigTemplate

            if __:
                print(f'\n*** BATCH SIZE LIST IS NOT PROPERLY SIZED.  MUST CONFIGURE BATCH SIZE OR INVOKE A STANDARD CONFIG TO LOAD')
                print(f'HYPERPARAMETERS AND FORCE BUILD OF BATCH SIZE LIST *** \n')
                self.sub_config = 'BYPASS'  # CHANGE FROM 'A', FORCE BYPASS OF 'A' IN MLConfigTemplate

            if ___:
                print(f'\n*** LEARNING RATE LIST IS NOT PROPERLY SIZED.  MUST CONFIGURE LEARNING RATES OR INVOKE A STANDARD CONFIG TO LOAD')
                print(f'HYPERPARAMETERS AND FORCE BUILD OF LEARNING RATE LIST ***\n')
                self.sub_config = 'BYPASS'  # CHANGE FROM 'A', FORCE BYPASS OF 'A' IN MLConfigTemplate

            if ____:
                print(f'\n*** TARGET IS NOT BINARY, COST FUNCTION MUST BE LEAST SQUARES AND FINAL LINK MUST BE NONE ***\n')
                self.sub_config = 'BYPASS'  # CHANGE FROM 'A', FORCE BYPASS OF 'A' IN MLConfigTemplate

            if True not in [_, __, ___, ____]:
                print(f'\n*** ARRAY OF NODES, BATCH SIZE, AND LEARNING RATES ARE CORRECTLY SIZED. ***\n')
                # DONT CHANGE self.sub_config FROM 'A', ALLOW TO HIT break UNDER 'A' IN MLConfigTemplate

            self.config_str = []  # FORCE BYPASS OF NNCoreConfigCode BELOW

        # DO COMPULSORY CONFIG (IF 'E' WAS CALLED IN THE PARENT)
        elif self.sub_config == 'E': self.config_str = 'FSRGHTWN'    # 6/10/23 BEAR WAS 'FSOGHKWN'
        # IF USER CHOSE STANDARD CONFIG, FORCE BUILD OF AON WITH LOADED CONFIG INFO, ALSO GIVES OPTION TO FILL AON FROM FILE
        elif self.sub_config == 'I': self.config_str = 'BR'  # FORCE BUILD OF "BATCH_SIZE" HERE, THEN BUILD AON (OTHERWISE
            # WOULD HAVE TO SEND "DATA" & MAYBE "OUTPUT_VECTOR" THRU THE STANDARD CONFIG MACHINERY, SIMPLER JUST TO BUILD
            # BATCH_SIZE MANUALLY HERE)
        else: self.config_str = self.sub_config         # OTHERWISE, JUST RUN THE sub_config AS SELECTED BY USER.


        for self.sub_config in self.config_str:
            self.ARRAY_OF_NODES, self.NEURONS, self.nodes, self.node_seed, self.activation_constant, \
            self.aon_base_path, self.aon_filename, self.cost_fxn, self.SELECT_LINK_FXN, self.LIST_OF_NN_ELEMENTS, \
            self.batch_method, self.BATCH_SIZE, self.gd_method, self.conv_method, self.lr_method, self.LEARNING_RATE, \
            self.momentum_weight, self.rglztn_type, self.rglztn_fctr, self.conv_kill, self.pct_change, self.conv_end_method, \
            self.gd_iterations, self.non_neg_coeffs, self.allow_summary_print, self.summary_print_interval, self.iteration = \
            nccc.NNCoreConfigCode(self.sub_config, self.DATA, self.TARGET_VECTOR, self.data_run_orientation,
                self.target_run_orientation, self.ARRAY_OF_NODES, self.NEURONS, self.nodes, self.node_seed, self.activation_constant,
                self.aon_base_path, self.aon_filename, self.cost_fxn, self.SELECT_LINK_FXN, self.LIST_OF_NN_ELEMENTS,
                self.OUTPUT_VECTOR, self.batch_method, self.BATCH_SIZE, self.gd_method, self.conv_method, self.lr_method,
                self.LEARNING_RATE, self.momentum_weight, self.rglztn_type, self.rglztn_fctr, self.conv_kill, self.pct_change,
                self.conv_end_method, self.gd_iterations, self.non_neg_coeffs, self.allow_summary_print, self.summary_print_interval,
                self.iteration).config()
        ####################################################################################################################
        ####################################################################################################################


    def print_parameters(self):
        _ = lambda text, width: str(text).ljust(width)
        __ = lambda text, width: str(text).center(width)
        xs = 6
        sh = 12
        lg = 60

        print()
        print(f"{_(' ',xs)}{__('LINK',sh)}{__(' ',sh)}{__('DESCENT',sh)}{__('LEARNING',lg)}{__('ACTIVATION',sh)}")
        print(f"{_('NODE',xs)}{__('FXN',sh)}{__('NEURONS',sh)}{__('METHOD',sh)}{__('RATE (FIRST 5)',lg)}{__('CONSTANT',sh)}")

        LR_DUM = [[] for _ in range(self.nodes)]   # LEARNING_RATE MODIFIER FOR DISPLAY
        for node_idx in range(self.nodes):
            try: LR_DUM[node_idx] = [f'{_:.5g}' for _ in self.LEARNING_RATE[node_idx]]
            except: pass

        for idx in range(self.nodes):
            print(
                f"{__(idx + 1, xs)}" + \
                f"{__(self.SELECT_LINK_FXN[idx], sh)}" + \
                f"{__(self.NEURONS[idx], sh)}" + \
                f"{__(dict({'G': 'GRADIENT', 'C': 'COORDINATE'})[self.gd_method], sh)}" + \
                f"{__(LR_DUM[idx][0:5], lg)}" + \
                f"{__(self.activation_constant, sh)}"
            )

        NN_PARAMS_PRINT = dict({
            f'cost_fxn': self.cost_fxn, # {ec.cost_functions()[self.cost_fxn],  BEAR
            f'gd_iterations': self.gd_iterations,
            f'gd_method': dict({'G': 'GRADIENT', 'C': 'COORDINATE'})[self.gd_method],
            f'conv_method': dict({'G': 'GRADIENT', 'R': 'RMSPROP', 'A': 'ADAM', 'N': 'NEWTON'})[self.conv_method],
            f'lr_method': dict({'C': 'CONSTANT', 'S':'CAUCHY'})[self.lr_method],
            f'batch_method': dict({'S': 'STOCASTIC', 'B': 'BATCH', 'M': 'MINI-BATCH'})[self.batch_method],
            f'regularization_type': self.rglztn_type,
            f'regularization_fctr': self.rglztn_fctr,
            f'momentum_weight': self.momentum_weight
        })

        NN_PARAMS_PRINT2 = dict({
            f'conv_kill': self.conv_kill,
            f'pct_change': self.pct_change,
            f'conv_end_method': self.conv_end_method,
            f'allow_summary_print': self.allow_summary_print,
            f'summary_print_interval': self.summary_print_interval,
            f'aon_base_path': self.aon_base_path,
            f'aon_filename': self.aon_filename,
            f'non_neg_coeffs': self.non_neg_coeffs,
            f'iteration': self.iteration
        })

        # CREATE COMBINED DICT FOR SYNCRONIZING PARSING OF PRINTOUTS
        DUM = NN_PARAMS_PRINT | NN_PARAMS_PRINT2
        max_key_len = max(map(len, DUM))
        max_value_len = max(map(lambda x: len(str(DUM[x])), NN_PARAMS_PRINT))

        print()
        _ = [str(k).ljust(max_key_len + 5) + str(v).ljust(max_value_len) for k,v in NN_PARAMS_PRINT.items()]
        __ = [str(k).ljust(max_key_len + 5) + str(v).ljust(max_value_len) for k,v in NN_PARAMS_PRINT2.items()]
        except_text = ' ' * (max_key_len + 5 + max_value_len)
        for line_idx in range(max(len(NN_PARAMS_PRINT), len(NN_PARAMS_PRINT2))):
            try: params1_txt = _[line_idx]
            except: params1_txt = except_text

            try: params2_txt = __[line_idx]
            except: params2_txt = except_text

            print(params1_txt + ' ' * 10 + params2_txt)

        del DUM, max_key_len, max_value_len, except_text, params1_txt, params2_txt


        print(f'\nBATCH_SIZE[:10] = {self.BATCH_SIZE[:10]}')


    def return_fxn(self):
        return self.ARRAY_OF_NODES, self.NEURONS, self.nodes, self.node_seed, self.activation_constant, \
               self.aon_base_path, self.aon_filename, self.cost_fxn, self.SELECT_LINK_FXN, self.LIST_OF_NN_ELEMENTS, \
               self.OUTPUT_VECTOR, self.batch_method, self.BATCH_SIZE, self.gd_method, self.conv_method, self.lr_method, \
               self.LEARNING_RATE, self.momentum_weight, self.rglztn_type, self.rglztn_fctr, self.conv_kill, \
               self.pct_change, self.conv_end_method, self.gd_iterations, self.non_neg_coeffs, self.allow_summary_print, \
               self.summary_print_interval, self.iteration















if __name__ == '__main__':
    pass






















