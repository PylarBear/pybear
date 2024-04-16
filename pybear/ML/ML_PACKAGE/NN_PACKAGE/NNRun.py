from copy import deepcopy
import numpy as n
from data_validation import validate_user_input as vui
from general_list_ops import list_select as ls
from ML_PACKAGE import MLRunTemplate as mlrt
from ML_PACKAGE.NN_PACKAGE import NNCoreRunCode as nncrc, NNConfig as nnc
from ML_PACKAGE.NN_PACKAGE.print_results import NNSummaryStatistics as nnss, AON_excel_dump as aed, nn_setup_dump as nnsd
    # nn_generic_test_results_dump as fd

from ML_PACKAGE.NN_PACKAGE.gd_run import output_vector_calc as ovc, error_calc as ec
from ML_PACKAGE.standard_configs import standard_configs as sc
from ML_PACKAGE.GENERIC_PRINT import print_post_run_options as ppro, print_vectors as pv, general_dev_results_dump as gdrd, \
                            general_test_results_dump as gtrd

# INHERITED ####################################################################################################################
# MODULE-SPECIFIC MAIN MENU DECLARATIONS ############################################################################
# END MODULE-SPECIFIC MAIN MENU DECLARATIONS ############################################################################

# GENERIC FUNCTIONS ####################################################################################################################
# row_column_display_select()            # user selects rows and columns for display on screen and filedump
# filedump_path()                        # hard-coded directory, user-entered filename
# filedump_general_ml_setup_module()     # module for filedump of general setup for all ML packages, used for train, dev, & test filedump
# dev_or_test_draw_params()              # module that returns the sample size to pull from train data for dev or test data sets
# random_dev_or_test_draw()              # randomly select user-specified quantity of examples for dev & test sets from (remaining) examples in TRAIN_SWNL
# partition_dev_or_test_draw()           # partition (remaining) examples in TRAIN_SWNL for selection into dev or test
# category_dev_or_test_draw()            # select examples for dev & test sets using categories in TRAIN_SWNL
# output_calc()                          # return module for running user-selected data into trained ML algorithm and calculating and sorting results for all ML packages, overwritten in child
# END GENERIC FUNCTIONS ####################################################################################################################

# TRAIN BUILD DEFINITIONS ################################################################################################
# reset_train_data()                     # reset train data to ORIGINAL DATA
# END TRAIN BUILD DEFINITIONS ################################################################################################

# DEV BUILD DEFINITIONS ##################################################################################################
# base_dev_build_module()                # module with code for building dev objects for all ML packages
# dev_build()                            # module for building dev objects, train objects starts as original objects, then dev objects are extracted from train objects
# END DEV BUILD DEFINITIONS ##################################################################################################

# TEST BUILD DEFINITIONS ##################################################################################################
# base_test_build_module()               # module with code for building test objects for all ML packages
# test_build()                           # module for building test objects, train objects starts as original objects, then test objects are extracted from train objects
# END TEST BUILD DEFINITIONS ##################################################################################################

# TRAIN CALC DEFINITIONS ###################################################################################################
# END TRAIN CALC DEFINITIONS ###################################################################################################

# DEV CALC DEFINITIONS ##################################################################################################
# rglztn_partition_iterator()            # only used to save space in base_dev_calc_module()
# base_dev_calc_module()                 # module for performing dev calculations for all ML packages
# END DEV CALC DEFINITIONS ##################################################################################################

# TEST CALC DEFINITIONS ##################################################################################################
# base_test_calc_module()                # module for performing test calculations for all ML packages
# END TEST CALC DEFINITIONS ##################################################################################################

# TRAIN DATA DISPLAY ##############################################################################################################
# END TRAIN DATA DISPLAY ##############################################################################################################

# DEV DATA DISPLAY ###########################################################################################################
# dev_summary_statistics_module()        # returns module for printing summary statistics of dev data for all ML packages
# print_dev_results_module()             # returns module for printing dev results to screen for all ML packages
# dev_filedump_module()                  # returns module for filedump of dev results for all ML packages
# END DEV DATA DISPLAY ############################################################################################A##############

# CALC DATA DISPLAY ############################################################################################A##############
# END CALC DATA DISPLAY ############################################################################################A##############

# base_return_fxn()                      # specify what to return for all ML packages
# run()                                  # MLRunTemplate loop



# OVERWRITTEN ####################################################################################################################
# MODULE-SPECIFIC MAIN MENU DECLARATIONS ############################################################################
#Xmodule_specific_main_menu_cmds()       # returns module-specific top-level menu options, overwritten in child
#Xmodule_specific_main_menu_operations() # returns module-specific execution code for post-run cmds, overwritten in child
# END MODULE-SPECIFIC MAIN MENU DECLARATIONS ############################################################################

# GENERIC FUNCTIONS ####################################################################################################################
#Xhyperparameter_display_module()        # print hyperparameter settings to screen
#Xfiledump_package_specific_setup_module()  # module for filedump of setup for specific package, used for train, dev, & test filedump, overwritten in child
#Xgeneric_ml_core_calc_algorithm()       # return module for package-specific core output algorithm, returns final output vector, overwritten in child
#Xtrain_data_calc_algorithm()            # module for passing train data thru package-specific core algorithm, returns final output vector
#Xdev_data_calc_algorithm()              # module for passing dev data thru package-specific core algorithm, returns final output vector
#Xtest_data_calc_algorithm()             # module for passing test data thru package-specific core algorithm, returns final output vector
#Xgeneric_ml_core_error_algorithm()      # return module for package-specific core error algorithm, returns total error, overwritten in child
#Xtrain_data_error_algorithm()           # module for passing train data thru package-specific error algorithm, returns total error
#Xdev_data_error_algorithm()             # module for passing dev data thru package-specific error algorithm, returns total error
#Xtest_data_error_algorithm()            # module for passing test data th
# END GENERIC FUNCTIONS ####################################################################################################################

# TRAIN BUILD DEFINITIONS #####################################################################################################
# END TRAIN BUILD DEFINITIONS ################################################################################################

# DEV BUILD DEFINITIONS ##################################################################################################
#Xsub_dev_build_module()                 # return module with package-specific dev objects build code, overwritten in child
#Xsub_dev_build_cmds()                   # return list with package-specific dev objects build prompts, overwritten in child
# END DEV BUILD DEFINITIONS ##################################################################################################

# TEST BUILD DEFINITIONS ##################################################################################################
#Xsub_test_build_module()                # return module with package-specific test objects build code, overwritten in child
#Xsub_test_build_cmds()                  # return list with package-specific test objects build prompts, overwritten in child
# END TEST BUILD DEFINITIONS ##################################################################################################

# TRAIN CALC DEFINITIONS ###################################################################################################
#Xcore_training_code()                   # return unique core training algorithm for particular ML package
#Xkfold_core_training_code()             # run code with randomization of nn parameters for k-fold
# END TRAIN CALC DEFINITIONS ###################################################################################################

# DEV CALC DEFINITIONS ##################################################################################################
#Xsub_dev_calc_module()                  # return module with package-specific commands for performing dev calculations
#Xsub_dev_calc_cmds()                    # return list with package-specific dev calc prompts
# END DEV CALC DEFINITIONS ##################################################################################################

# TEST CALC DEFINITIONS ##################################################################################################
#Xsub_test_calc_module()                 # return module with package-specific commands for performing test calculations
#Xsub_test_calc_cmds()                   # return list with package-specific test calc prompts
# END TEST CALC DEFINITIONS ##################################################################################################

# TRAIN DATA DISPLAY ##############################################################################################################
#Xtrain_summary_statistics_module()      # returns module for printing summary statistics of train data for particular ML package
#Xprint_train_results_module()           # returns module for printing train results to screen for particular ML package
#Xtrain_filedump_module()                # returns module for filedump of train results for particular ML package
# END TRAIN DATA DISPLAY ##############################################################################################################

# DEV DATA DISPLAY ###########################################################################################################
# END DEV DATA DISPLAY ###########################################################################################################

# CALC DATA DISPLAY ###########################################################################################################
#Xtest_summary_statistics_module()       # returns module for printing summary statistics of test data for particular ML package
#Xprint_test_results_module()            # returns module for printing test results to screen for particular ML package
#Xtest_filedump_module()                 # returns module for filedump of test results for particular ML package
# END CALC DATA DISPLAY ###########################################################################################################

#Xreturn_fxn()                           # return self.base_return_fxn()




class NNRun(mlrt.MLRunTemplate):

    def __init__(self, standard_config, nn_config, SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS, SUPER_WORKING_NUMPY_LIST, WORKING_SUPOBJS,
        data_run_orientation, target_run_orientation, refvecs_run_orientation, WORKING_CONTEXT, WORKING_KEEP, TRAIN_SWNL,
        DEV_SWNL, TEST_SWNL, split_method, LABEL_RULES, number_of_labels, event_value, negative_value, conv_kill, pct_change,
        conv_end_method, rglztn_type, rglztn_fctr, bypass_validation, ARRAY_OF_NODES, NEURONS, LIST_OF_NN_ELEMENTS,
        SELECT_LINK_FXN, BATCH_SIZE, LEARNING_RATE, OUTPUT_VECTOR, nodes, node_seed, new_error_start, aon_base_path,
        aon_filename, activation_constant, gd_iterations, cost_fxn, allow_summary_print, summary_print_interval, batch_method,
        gd_method, conv_method, lr_method, momentum_weight, non_neg_coeffs, iteration):


        super().__init__(standard_config, nn_config, SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS, SUPER_WORKING_NUMPY_LIST,
            WORKING_SUPOBJS, data_run_orientation, target_run_orientation, refvecs_run_orientation, WORKING_CONTEXT,
            WORKING_KEEP, TRAIN_SWNL, DEV_SWNL, TEST_SWNL, split_method, LABEL_RULES, number_of_labels, event_value,
            negative_value, conv_kill, pct_change, conv_end_method, rglztn_type, rglztn_fctr, bypass_validation, __name__)


        self.ARRAY_OF_NODES = ARRAY_OF_NODES
        self.NEURONS = NEURONS
        self.LIST_OF_NN_ELEMENTS = LIST_OF_NN_ELEMENTS
        self.SELECT_LINK_FXN = SELECT_LINK_FXN
        self.BATCH_SIZE = BATCH_SIZE
        self.LEARNING_RATE = LEARNING_RATE
        self.LEARNING_RATE_BACKUP = deepcopy(self.LEARNING_RATE)   # NEED THIS TO RESET LEARNING RATES BEFORE EACH RUN IN K-FOLD
        self.TRAIN_OUTPUT_VECTOR = OUTPUT_VECTOR
        self.nodes = nodes
        self.node_seed = node_seed
        self.new_error_start = new_error_start
        self.aon_base_path = aon_base_path
        self.aon_filename = aon_filename
        self.activation_constant = activation_constant
        self.gd_iterations = gd_iterations
        self.cost_fxn = cost_fxn.upper()
        self.allow_summary_print = allow_summary_print
        self.summary_print_interval = summary_print_interval
        self.batch_method = batch_method
        self.gd_method = gd_method
        self.conv_method = conv_method
        self.lr_method = lr_method
        self.momentum_weight = momentum_weight
        self.non_neg_coeffs = non_neg_coeffs
        self.early_stop_interval = 1e12
        self.iteration = iteration


    #  INHERITS FOR NOW ####################################################################################################
    # MODULE-SPECIFIC MAIN MENU DECLARATIONS ############################################################################
    def module_specific_main_menu_cmds(self):
        # module-specific top-level menu options
        # SPECIFIED IN CHILDREN   # CANT USE 'ABCDEFGHIJKLNOPQRSTUVWXZ'
        return {
                '0': '\n\ndump AON elements to file',
                '1': 'set regularization factor',
                '2': 'set regularization type',
        }


    def module_specific_main_menu_operations(self):
        # execution code for post-run cmds          # CANT USE 'ABCDEFGHIJKLNOPQRSTUVWXZ'
        if self.post_run_select == '0':
            print(f'\nWrites AON elements to file.')
            print(f'\nAON values originally loaded from {self.aon_base_path + self.aon_filename}')
            aed.AON_excel_dump(self.ARRAY_OF_NODES)
            # THIS IS SEPARATE FROM ALL OTHER FILE DUMPS TO ENABLE EASY LOADING OF NN PARAMETERS
        # THIS IS HERE TO ALLOW CHANGE WITHOUT HAVING TO GO BACK TO ConfigRun (WHICH PRESERVES self.DEV_ERROR)
        if self.post_run_select == '1':
            self.rglztn_fctr = vui.validate_user_float(f'Enter regularization factor (currently {self.rglztn_fctr}) > ', min=0)
        if self.post_run_select == '2':
            self.rglztn_type = {1:'L1', 2:'L2'}[
                vui.validate_user_int(f'Select L1(1) or L2(2) regularization (currently {self.rglztn_type}) > ', min=1, max=2)]

    # END MODULE-SPECIFIC MAIN MENU DECLARATIONS ############################################################################


    # OVERWRITES METHODS #######################################################################################3
    # GENERIC FUNCTIONS ####################################################################################################################
    def hyperparameter_display_module(self):
        # print hyperparameter settings to screen
        print(f'\nNEURAL NETWORK / REGRESSION HYPERPARAMETER SETTINGS:')
        nnc.NNConfig(self.standard_config, self.sub_config, self.SUPER_WORKING_NUMPY_LIST, self.WORKING_SUPOBJS,
            self.data_run_orientation, self.target_run_orientation, self.ARRAY_OF_NODES, self.NEURONS, self.nodes,
            self.node_seed, self.activation_constant, self.aon_base_path, self.aon_filename, self.cost_fxn,
            self.SELECT_LINK_FXN, self.LIST_OF_NN_ELEMENTS, self.TRAIN_OUTPUT_VECTOR, self.batch_method, self.BATCH_SIZE,
            self.gd_method, self.conv_method, self.lr_method, self.LEARNING_RATE, self.momentum_weight, self.rglztn_type,
            self.rglztn_fctr, self.conv_kill, self.pct_change, self.conv_end_method, self.gd_iterations, self.non_neg_coeffs,
            self.allow_summary_print, self.summary_print_interval, self.iteration, self.bypass_validation).print_parameters()


    '''def row_column_display_select(self):
        INHERITED '''

    '''def filedump_general_ml_setup_module(self):
        INHERITED'''

    def filedump_package_specific_setup_module(self):
        # module for filedump of setup for specific package, used for train, dev, & test filedump, overwritten in child
        self.wb = nnsd.nn_setup_dump(self.wb, self.NEURONS, self.nodes, self.activation_constant, self.aon_base_path,
            self.aon_filename, self.cost_fxn, self.SELECT_LINK_FXN, self.batch_method, self.BATCH_SIZE, self.gd_method,
            self.conv_method, self.lr_method, self.LEARNING_RATE, self.momentum_weight, self.rglztn_type, self.rglztn_fctr,
            self.conv_kill, self.pct_change, self.conv_end_method, self.gd_iterations, self.non_neg_coeffs,
            self.allow_summary_print, self.summary_print_interval, self.iteration)

    '''def dev_or_test_draw_params(self):
        INHERITED'''

    '''def random_dev_or_test_draw(self, TRAIN_SWNL, size):
        INHERITED'''

    '''def partition_dev_or_test_draw(self, TRAIN_SWNL, number_of_partitions, partition_number):
        INHERITED'''

    '''def category_dev_or_test_draw():
        INHERITED'''

    def generic_ml_core_calc_algorithm(self):
        # return module for package-specific core output algorithm, returns final output vector, overwritten in child
        # not in use
        pass


    def train_data_calc_algorithm(self):
        # module for passing train data thru package-specific core algorithm, returns final output vector
        return ovc.output_vector_calc(self.TRAIN_SWNL[0], self.ARRAY_OF_NODES, self.SELECT_LINK_FXN,
                                                          [], self.activation_constant)


    def dev_data_calc_algorithm(self):
        # module for passing dev data thru package-specific core algorithm, returns final output vector
        return ovc.output_vector_calc(self.DEV_SWNL[0], self.ARRAY_OF_NODES, self.SELECT_LINK_FXN,
                                                          [], self.activation_constant)


    def test_data_calc_algorithm(self):
        # module for passing test data thru package-specific core algorithm, returns final output vector
        return ovc.output_vector_calc(self.TEST_SWNL[0], self.ARRAY_OF_NODES, self.SELECT_LINK_FXN,
                                                        [], self.activation_constant)


    def generic_ml_core_error_algorithm(self):
        # module for package-specific core error algorithm, returns total error
        # not in use
        pass


    def train_data_error_algorithm(self):
        # module for passing train data thru package-specific error algorithm, returns total error
        self.TRAIN_OUTPUT_VECTOR = ovc.output_vector_calc(self.TRAIN_SWNL[0], self.ARRAY_OF_NODES, self.SELECT_LINK_FXN,
                                                          [], self.activation_constant)

        return ec.error_calc(self.ARRAY_OF_NODES, self.TRAIN_SWNL[1], self.TRAIN_OUTPUT_VECTOR, self.cost_fxn, 0,
                   self.SELECT_LINK_FXN, self.rglztn_type, 0)
        # 3-5-22 CHANGED LAST TERM FROM self.rglztn_fctr TO 0, THINKING SHOULDNT INCLUDE LENGTH OUTSIDE OF TRAINING
        # core_training_code MUST HAVE rglztn_fctr


    def dev_data_error_algorithm(self):
        # module for passing dev data thru package-specific error algorithm, returns total error
        self.DEV_OUTPUT_VECTOR = ovc.output_vector_calc(self.DEV_SWNL[0], self.ARRAY_OF_NODES, self.SELECT_LINK_FXN,
                                                          [], self.activation_constant)
        return ec.error_calc(self.ARRAY_OF_NODES, self.DEV_SWNL[1], self.DEV_OUTPUT_VECTOR, self.cost_fxn, 0,
                   self.SELECT_LINK_FXN, self.rglztn_type, 0)
        # 3-5-22 CHANGED LAST TERM FROM self.rglzth_fctr TO 0, THINKING SHOULDNT INCLUDE LENGTH OUTSIDE OF TRAINING


    def test_data_error_algorithm(self):
        # module for passing test data thru package-specific error algorithm, returns total error
        self.TEST_OUTPUT_VECTOR = ovc.output_vector_calc(self.TEST_SWNL[0], self.ARRAY_OF_NODES, self.SELECT_LINK_FXN,
                                                          [], self.activation_constant)
        return ec.error_calc(self.ARRAY_OF_NODES, self.TEST_SWNL[1], self.TEST_OUTPUT_VECTOR, self.cost_fxn, 0,
                   self.SELECT_LINK_FXN, self.rglztn_type, 0)
        # 3-5-22 CHANGED LAST TERM FROM self.rglzth_fctr TO 0, THINKING SHOULDNT INCLUDE LENGTH OUTSIDE OF TRAINING


    '''def output_calc(self, calc_algorithm, CALC_OBJECT, object_name):
        # return module with commands to run user-selected matrix for all ML packages
        INHERITED'''


    # END GENERIC FUNCTIONS ####################################################################################################################

    # TRAIN BUILD DEFINITIONS ################################################################################################
    '''def reset_train_data(self):  # reset train data to ORIGINAL DATA
        INHERITED'''
    # END TRAIN BUILD DEFINITIONS ################################################################################################

    # DEV BUILD DEFINITIONS ##################################################################################################
    '''def base_dev_build_module(self):
        INHERITED'''


    def sub_dev_build_module(self):
        # return module with package-specific test matrix build commands, overwritten in child
        # overwritten in child
        pass


    def sub_dev_build_cmds(self):
        # return list with package-specific test matrix build commands
        return {}  # SPECIFIED IN CHILDREN   CANT USE 'RSDFTUVBNA'


    '''def dev_build(self):
        INHERITED'''

    # END DEV BUILD DEFINITIONS ##################################################################################################

    # TEST BUILD DEFINITIONS ##################################################################################################
    '''def base_test_build_module(self):
        INHERITED'''


    def sub_test_build_module(self):
        # return module with package-specific test matrix build commands, overwritten in child
        # overwritten in child
        pass


    def sub_test_build_cmds(self):
        # return list with package-specific test matrix build commands
        return {}  # SPECIFIED IN CHILDREN   CANT USE 'DRSPFTUBONA'


    '''def test_build(self):
        INHERITED'''


    # END TEST BUILD DEFINITIONS ##################################################################################################

    # TRAIN CALC DEFINITIONS ###################################################################################################
    def core_training_code(self):
        # unique run code for particular ML package

        print(f'\nResetting learning rates...')
        self.LEARNING_RATE = deepcopy(self.LEARNING_RATE_BACKUP)
        print(f'Done.')

        if n.array_equiv(self.DEV_SWNL, []):
            self.ARRAY_OF_NODES, self.TRAIN_OUTPUT_VECTOR, self.iteration = \
            nncrc.NNCoreRunCode(self.TRAIN_SWNL[0], self.TRAIN_SWNL[1], self.data_run_orientation, self.target_run_orientation,
                self.ARRAY_OF_NODES, self.TRAIN_OUTPUT_VECTOR, self.SELECT_LINK_FXN, self.BATCH_SIZE, self.LEARNING_RATE,
                self.LIST_OF_NN_ELEMENTS, self.new_error_start, self.cost_fxn, self.batch_method, self.gd_method, self.conv_method,
                self.lr_method, self.momentum_weight, self.rglztn_type, self.rglztn_fctr, self.conv_kill, self.pct_change,
                self.conv_end_method, self.activation_constant, self.gd_iterations, self.non_neg_coeffs, self.allow_summary_print,
                self.summary_print_interval, self.iteration).run()
        else:
            self.ARRAY_OF_NODES, self.TRAIN_OUTPUT_VECTOR, self.iteration = \
            nncrc.NNCoreRunCode_MLPackage(self.TRAIN_SWNL[0], self.TRAIN_SWNL[1], self.DEV_SWNL[0], self.DEV_SWNL[1],
                self.data_run_orientation, self.target_run_orientation, self.ARRAY_OF_NODES, self.TRAIN_OUTPUT_VECTOR,
                self.SELECT_LINK_FXN, self.BATCH_SIZE, self.LEARNING_RATE, self.LIST_OF_NN_ELEMENTS, self.new_error_start,
                self.cost_fxn, self.batch_method, self.gd_method, self.conv_method, self.lr_method, self.momentum_weight,
                self.rglztn_type, self.rglztn_fctr, self.conv_kill, self.pct_change, self.conv_end_method, self.activation_constant,
                self.gd_iterations, self.non_neg_coeffs, self.allow_summary_print, self.summary_print_interval,
                self.early_stop_interval, self.iteration).run()


    def kfold_core_training_code(self):
        # run code with randomization of nn parameters for k-fold
        # RANDOMIZATION OF NN PARAMS AFTER EVERY TRAIN FOR NEXT TRAIN
        self.iteration = 0  # MAKE SURE WHEN DOING K-FOLD TO START ITER AT 0, FOR CORRECT LEARNING RATES, NOT SURE IF THIS IS NEEDED HERE
        
        print(f'\nRandomly seeding NN params...')
        for layer_idx in range(len(self.ARRAY_OF_NODES)):
            for node_idx in range(len(self.ARRAY_OF_NODES[layer_idx])):
                for param_idx in range(len(self.ARRAY_OF_NODES[layer_idx][node_idx])):
                    self.ARRAY_OF_NODES[layer_idx][node_idx][param_idx] = \
                        n.random.normal(1e-4, 1e-1)  # MEAN, STDEV
        print(f'Done.')

        print(f'\nResetting learning rates...')
        self.LEARNING_RATE = deepcopy(self.LEARNING_RATE_BACKUP)
        print(f'Done.')

        self.ARRAY_OF_NODES, self.TRAIN_OUTPUT_VECTOR, self.iteration = \
            nncrc.NNCoreRunCode_MLPackage(self.TRAIN_SWNL[0], self.TRAIN_SWNL[1], self.DEV_SWNL[0], self.DEV_SWNL[1],
                self.data_run_orientation, self.target_run_orientation, self.ARRAY_OF_NODES, self.TRAIN_OUTPUT_VECTOR,
                self.SELECT_LINK_FXN, self.BATCH_SIZE, self.LEARNING_RATE, self.LIST_OF_NN_ELEMENTS, self.new_error_start,
                self.cost_fxn, self.batch_method, self.gd_method, self.conv_method, self.lr_method, self.momentum_weight,
                self.rglztn_type, self.rglztn_fctr, self.conv_kill, self.pct_change, self.conv_end_method,
                self.activation_constant, self.gd_iterations, self.non_neg_coeffs, self.allow_summary_print,
                self.summary_print_interval, self.early_stop_interval, self.iteration).run()


    # END TRAIN CALC DEFINITIONS ###################################################################################################

    # DEV CALC DEFINITIONS ##################################################################################################

    '''def rglztn_partition_iterator(self, number_of_partitions):
        INHERITED'''


    ''''def base_dev_calc_module(self):
        INHERITED'''


    def sub_dev_calc_module(self):
        # return module with package-specific commands to run current dev matrix
        # overwritten in child
        pass


    def sub_dev_calc_cmds(self):
        # return list with package - specific dev calc commands
        return {}  # SPECIFIED IN CHILDREN   CANT USE 'DSA'

    # END DEV CALC DEFINITIONS ##################################################################################################

    # TEST CALC DEFINITIONS ##################################################################################################
    '''def base_test_calc_module(self):
        INHERITED'''


    def sub_test_calc_module(self):
        # return module with package-specific commands to run current test matrix
        # 3-5-22 BEAR FIX
        if self.test_calc_select == 'S':
            self.CSUTM_DF = sc.test_cases_calc_standard_configs(self.standard_config, self.TEST_SWNL[0],
                self.TRAIN_SWNL[0], self.ARRAY_OF_NODES, self.SELECT_LINK_FXN, self.TRAIN_SWNL[1],
                self.activation_constant)


    def sub_test_calc_cmds(self):
        # return list with package - specific test calc commands
        return {'s': 'run special NN test from standard configs'}                       # SPECIFIED IN CHILDREN   CANT USE 'NA'


    # END TEST CALC DEFINITIONS ##################################################################################################

    # TRAIN DATA DISPLAY ##############################################################################################################

    def train_summary_statistics_module(self):
        # returns module for printing summary statistics of train data
        # BEAR FIX
        error = self.train_data_error_algorithm()  # DO THIS OUT HERE TO ENSURE self.TRAIN_OUTPUT_VECTOR IS UPDATED B4 nnss
        nnss.NNSummaryStatisticsPrint(self.TRAIN_OUTPUT_VECTOR, self.TRAIN_SWNL[1], self.SELECT_LINK_FXN[-1],
                                 error, self.ARRAY_OF_NODES, self.iteration + 1, self.gd_iterations).print()


    def print_train_results_module(self):
        # returns module for printing train results for particular ML package
        error = self.train_data_error_algorithm()  # DO THIS OUT HERE TO ENSURE self.TRAIN_OUTPUT_VECTOR IS UPDATED B4 nnss
        nnss.NNSummaryStatisticsPrint(self.TRAIN_OUTPUT_VECTOR, self.TRAIN_SWNL[1], self.SELECT_LINK_FXN[-1],
                                 error, self.ARRAY_OF_NODES, self.iteration + 1, self.gd_iterations).print()


    def train_filedump_module(self):
        # returns module for filedump of train results for particular ML package
        self.filedump_general_ml_setup_module()
        self.wb = nnsd.nn_setup_dump(self.wb, self.NEURONS, self.nodes,  self.activation_constant, self.aon_base_path,
                self.aon_filename, self.cost_fxn, self.SELECT_LINK_FXN, self.batch_method, self.BATCH_SIZE,
                self.gd_method, self.conv_method, self.lr_method, self.LEARNING_RATE, self.momentum_weight,
                self.rglztn_type, self.rglztn_fctr, self.conv_kill, self.pct_change, self.conv_end_method, self.gd_iterations,
                self.non_neg_coeffs, self.allow_summary_print, self.summary_print_interval, self.iteration)

        # self.wb = gtrd.nn_train_results_dump(self.wb, self.TRAIN_RESULTS)   NOT IN USE 4-2-22

        error = self.train_data_error_algorithm()  # PUT THIS HERE TO ENSURE self.TRAIN_OUTPUT_VECTOR IS UPDATED
        self.wb = nnss.NNSummaryStatisticsDump(self.wb, self.TRAIN_OUTPUT_VECTOR, self.TRAIN_SWNL[1], self.SELECT_LINK_FXN[-1],
                        error, 'TRAIN STATISTICS', self.ARRAY_OF_NODES, self.iteration+1, self.gd_iterations).dump()

    # END TRAIN DATA DISPLAY ##############################################################################################################

    # DEV DATA DISPLAY ##############################A##############################A##############################A##############
    def dev_summary_statistics_module(self):
        # returns module for printing summary statistics of dev data
        nnss.NNSummaryStatisticsPrint(self.DEV_OUTPUT_VECTOR, self.DEV_SWNL[1], self.SELECT_LINK_FXN[-1],
            self.new_error_start, self.ARRAY_OF_NODES, self.iteration + 1, self.gd_iterations).print()


    '''def print_dev_results_module(self):
        # returns module for printing dev results to screen for all ML packages
        pass
        INHERITED'''


    def dev_filedump_module(self):
        # returns module for filedump of dev results for all ML packages
        # DEV OBJECTS OTHER THAN self.DEV_ERROR MAY NOT EXIST IF DID K-FOLD CV
        self.wb = gdrd.general_dev_results_dump(self.wb, self.DEV_ERROR, self.RGLZTN_FACTORS)

        try:
            error = self.dev_data_error_algorithm()  # DO THIS FIRST TO ENSURE self.DEV_OUTPUT_VECTOR IS UPDATED B4 nnss
            self.wb = nnss.NNSummaryStatisticsDump(self.wb, self.DEV_OUTPUT_VECTOR, self.DEV_SWNL[1], self.SELECT_LINK_FXN[-1],
                    error, 'DEV STATISTICS', self.ARRAY_OF_NODES, self.iteration+1, self.gd_iterations).dump()
        except: pass

    # END DEV DATA DISPLAY ##############################A##############################A##############################A##############

    # CALC DATA DISPLAY ##############################A##############################A##############################A##############
    def test_summary_statistics_module(self):
        # returns module for printing summary statistics of test data
        error = self.test_data_error_algorithm()   # DO THIS OUT HERE TO ENSURE self.TEST_OUTPUT_VECTOR IS UPDATED B4 nnss
        nnss.NNSummaryStatisticsPrint(self.TEST_OUTPUT_VECTOR, self.TEST_SWNL[1], self.SELECT_LINK_FXN[-1],
                                 error, self.ARRAY_OF_NODES, self.iteration + 1, self.gd_iterations).print()


    '''
    def print_test_results_module(self):
        pass
        INHERITED
        '''


    def test_filedump_module(self):
        # package-specific module for saving test results
        self.wb = gtrd.general_test_results_dump(self.wb, self.CSUTM_DF, self.DISPLAY_COLUMNS, self.display_select, self.display_rows)
        error = self.test_data_error_algorithm()  # PUT THIS HERE TO ENSURE self.TEST_OUTPUT_VECTOR IS UPDATED
        self.wb = nnss.NNSummaryStatisticsDump(self.wb, self.TEST_OUTPUT_VECTOR, self.TEST_SWNL[1], self.SELECT_LINK_FXN[-1],
                    error, 'TEST STATISTICS', self.ARRAY_OF_NODES, self.iteration+1, self.gd_iterations).dump()

        # fd.nn_generic_test_results_dump('GD', self.standard_config, self.sub_config, self.SUPER_RAW_NUMPY_LIST,
        #         self.SUPER_WORKING_NUMPY_LIST, self.WORKING_VALIDATED_DATATYPES, self.WORKING_MODIFIED_DATATYPES,
        #         self.WORKING_FILTERING, self.WORKING_MIN_CUTOFFS, self.WORKING_USE_OTHER, self.WORKING_CONTEXT,
        #         self.WORKING_KEEP, self.split_method, self.LABEL_RULES, self.number_of_labels, self.event_value,
        #         self.negative_value, self.gd_method, self.conv_method, self.lr_method, self.LEARNING_RATE,
        #         self.SELECT_LINK_FXN, self.non_neg_coeffs, self.ROWID_VECTOR, self.TEST_OUTPUT_VECTOR, self.ARRAY_OF_NODES,
        #         self.CSUTM_DF, self.display_select, self.display_rows)

    # END CALC DATA DISPLAY ##############################A##############################A##############################A##############

    '''def base_return_fxn(self):
        INHERITED'''


    def return_fxn(self):

        return *self.base_return_fxn(), self.ARRAY_OF_NODES, self.TRAIN_OUTPUT_VECTOR, self.iteration, self.rglztn_fctr


    '''def run(self):  # MLRunTemplate loop
        INHERITED'''





if __name__ == '__main__':

    import numpy as n

    standard_config = ''
    nn_config = ''
    RAW_DATA = n.array([
        ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B'],
        ['A', 'B', 'C', 'B', 'C', 'A', 'C', 'A']
    ], dtype=object)
    RAW_DATA_HEADER = n.array([['AA','BB']], dtype=object)
    RAW_TARGET = n.array([[0,1,0,0,1,1,0,1]], dtype=object)
    RAW_TARGET_HEADER = n.array([['TARGET']], dtype=object)
    RAW_REF_VECS = n.array([[*range(1,9)]], dtype=object)
    RAW_REF_VECS_HEADER = n.array([['ROW_ID']], dtype=object)
    RAW_TEST_MATRIX = n.array([['DUM', 'DUM','DUM','DUM','DUM','DUM','DUM','DUM']], dtype=object)
    RAW_TEST_MATRIX_HEADER = n.array([['DUM']], dtype=object)
    SUPER_RAW_NUMPY_LIST = n.array([RAW_DATA, RAW_DATA_HEADER, RAW_TARGET, RAW_TARGET_HEADER, RAW_REF_VECS, RAW_REF_VECS_HEADER,
                            RAW_TEST_MATRIX, RAW_TEST_MATRIX_HEADER], dtype=object)

    #[[] = COLUMNS]
    WORKING_DATA = n.array([
        [1,0,0,1,0,0,1,0],
        [0,1,0,0,1,0,0,1],
        [0,0,1,0,0,1,0,0],
        [1,0,0,0,0,1,0,1],
        [0,1,0,1,0,0,0,0],
        [0,0,1,0,1,0,1,0]
    ], dtype=object)
    WORKING_DATA_HEADER = n.array([['AA - 1','AA - 2','AA - 3','BB - 1','BB - 2','BB - 3']], dtype=object)
    WORKING_TARGET = n.array([[0,1,0,0,1,1,0,1]], dtype=object)
    WORKING_TARGET_HEADER = n.array([['TARGET']], dtype=object)
    WORKING_REF_VECS = n.array([['DUM', 'DUM','DUM','DUM','DUM','DUM','DUM','DUM']], dtype=object)
    WORKING_REF_VECS_HEADER = n.array([['DUM']], dtype=object)
    WORKING_TEST_MATRIX = n.array([['DUM', 'DUM','DUM','DUM','DUM','DUM','DUM','DUM']], dtype=object)
    WORKING_TEST_MATRIX_HEADER = n.array([['DUM']], dtype=object)
    SUPER_WORKING_NUMPY_LIST = n.array([WORKING_DATA, WORKING_DATA_HEADER, WORKING_TARGET, WORKING_TARGET_HEADER, WORKING_REF_VECS,
                                WORKING_REF_VECS_HEADER, WORKING_TEST_MATRIX, WORKING_TEST_MATRIX_HEADER], dtype=object)

    WORKING_VALIDATED_DATATYPES = [['STR','STR','STR','STR','STR','STR'], '', ['STR'], '', ['STR'], '', ['STR'], '']
    WORKING_MODIFIED_DATATYPES = [['BIN','BIN','BIN','BIN','BIN','BIN'], '', ['STR'], '', ['STR'], '', ['STR'], '']
    WORKING_FILTERING = [[[],[],[],[],[],[]], '', [[]], '', [[]], '', [[]], '']
    WORKING_MIN_CUTOFFS = [[0,0,0,0,0,0], '', [0], '', [0], '', [0], '']
    WORKING_USE_OTHER = [['N','N','N','N','N','N'], '', ['N'], '', ['N'], '', ['N'], '']
    WORKING_CONTEXT = []
    WORKING_KEEP = ['A - 1','A - 2','A - 3','B - 1','B - 2','B - 3']
    WORKING_SCALING = ['','','','','','']

    TRAIN_DATA = n.array([
        [1,0,0,1,0],
        [0,1,0,0,1],
        [0,0,1,0,0],
        [1,0,0,0,0],
        [0,1,0,1,0],
        [0,0,1,0,1]
    ], dtype=object)
    TRAIN_DATA_HEADER = n.array([['AA - 1','AA - 2','AA - 3','BB - 1','BB - 2','BB - 3']], dtype=object)
    TRAIN_TARGET = n.array([[0,1,0,0,1]], dtype=object)
    TRAIN_TARGET_HEADER = n.array([['TARGET']], dtype=object)
    TRAIN_REF_VECS = n.array([['DUM', 'DUM','DUM','DUM','DUM','DUM','DUM','DUM']], dtype=object)
    TRAIN_REF_VECS_HEADER = n.array([['DUM']], dtype=object)
    TRAIN_TEST_MATRIX = n.array([['DUM', 'DUM','DUM','DUM','DUM','DUM','DUM','DUM']], dtype=object)
    TRAIN_TEST_MATRIX_HEADER = n.array([['DUM']], dtype=object)
    TRAIN_SWNL = n.array([WORKING_DATA, WORKING_DATA_HEADER, WORKING_TARGET, WORKING_TARGET_HEADER, WORKING_REF_VECS,
                                WORKING_REF_VECS_HEADER, WORKING_TEST_MATRIX, WORKING_TEST_MATRIX_HEADER], dtype=object)

    DEV_DATA = n.array([
        [0,1,0],
        [0,0,1],
        [1,0,0],
        [1,0,1],
        [0,0,0],
        [0,1,0]
    ], dtype=object)
    DEV_DATA_HEADER = n.array([['AA - 1','AA - 2','AA - 3','BB - 1','BB - 2','BB - 3']], dtype=object)
    DEV_TARGET = n.array([[1,0,1]], dtype=object)
    DEV_TARGET_HEADER = n.array([['TARGET']], dtype=object)
    DEV_REF_VECS = n.array([['DUM', 'DUM','DUM','DUM','DUM','DUM','DUM','DUM']], dtype=object)
    DEV_REF_VECS_HEADER = n.array([['DUM']], dtype=object)
    DEV_TEST_MATRIX = n.array([['DUM', 'DUM','DUM','DUM','DUM','DUM','DUM','DUM']], dtype=object)
    DEV_TEST_MATRIX_HEADER = n.array([['DUM']], dtype=object)
    DEV_SWNL = n.array([DEV_DATA, DEV_DATA_HEADER, DEV_TARGET, DEV_TARGET_HEADER, DEV_REF_VECS,
                                DEV_REF_VECS_HEADER, DEV_TEST_MATRIX, DEV_TEST_MATRIX_HEADER], dtype=object)

    TEST_DATA = n.array([
        [1,0,0,1,0,0,1,0],
        [0,1,0,0,1,0,0,1],
        [0,0,1,0,0,1,0,0],
        [1,0,0,0,0,1,0,1],
        [0,1,0,1,0,0,0,0],
        [0,0,1,0,1,0,1,0]
    ], dtype=object)
    TEST_DATA_HEADER = n.array([['AA - 1','AA - 2','AA - 3','BB - 1','BB - 2','BB - 3']], dtype=object)
    TEST_TARGET = n.array([[0,1,0,0,1,1,0,1]], dtype=object)
    TEST_TARGET_HEADER = n.array([['TARGET']], dtype=object)
    TEST_REF_VECS = n.array([['DUM', 'DUM','DUM','DUM','DUM','DUM','DUM','DUM']], dtype=object)
    TEST_REF_VECS_HEADER = n.array([['DUM']], dtype=object)
    TEST_TEST_MATRIX = n.array([['DUM', 'DUM','DUM','DUM','DUM','DUM','DUM','DUM']], dtype=object)
    TEST_TEST_MATRIX_HEADER = n.array([['DUM']], dtype=object)
    TEST_SWNL = n.array([TEST_DATA, TEST_DATA_HEADER, TEST_TARGET, TEST_TARGET_HEADER, TEST_REF_VECS,
                               TEST_REF_VECS_HEADER, TEST_TEST_MATRIX, TEST_TEST_MATRIX_HEADER], dtype=object)

    split_method = 'None'
    LABEL_RULES = []
    number_of_labels = 1
    event_value = 1
    negative_value = 0
    conv_kill = 100
    pct_change = 0.1
    conv_end_method = 'KILL'
    rglztn_type = 'L2'
    rglztn_fctr = 1
    ARRAY_OF_NODES = [
                        n.random.rand(3,6),
                        [[0.1, 0.01, -0.05]]
    ]
    NEURONS = 2
    LIST_OF_NN_ELEMENTS = []
    SELECT_LINK_FXN = ['Logistic', 'Logistic']
    BATCH_SIZE = 5
    OUTPUT_VECTOR = []
    nodes = 2
    node_seed = 0
    new_error_start = 0
    aon_base_path = ''
    aon_filename = ''
    activation_constant = 0
    gd_iterations = 10
    LEARNING_RATE = [[.01 for _ in range(gd_iterations)], [.001 for _ in range(gd_iterations)]]
    cost_fxn = 'L'
    allow_summary_print = 'Y'
    summary_print_interval = 200
    batch_method = 'B'
    gd_method = 'G'
    conv_method = 'A'
    lr_method = 'C'
    momentum_weight = 0.7
    non_neg_coeffs = 'N'
    iteration = 0

    TRAIN_SWNL, DEV_SWNL, TEST_SWNL, ARRAY_OF_NODES, TRAIN_OUTPUT_VECTOR, iteration, rglztn_fctr = \
        NNRun(standard_config, nn_config, SUPER_RAW_NUMPY_LIST, SUPER_WORKING_NUMPY_LIST, WORKING_VALIDATED_DATATYPES,
             WORKING_MODIFIED_DATATYPES, WORKING_FILTERING, WORKING_MIN_CUTOFFS, WORKING_USE_OTHER, WORKING_CONTEXT,
             WORKING_KEEP, WORKING_SCALING, TRAIN_SWNL, DEV_SWNL, TEST_SWNL, split_method, LABEL_RULES, number_of_labels,
             event_value, negative_value, conv_kill, pct_change, conv_end_method, rglztn_type, rglztn_fctr,
             ARRAY_OF_NODES, NEURONS, LIST_OF_NN_ELEMENTS, SELECT_LINK_FXN, BATCH_SIZE, LEARNING_RATE, OUTPUT_VECTOR, nodes,
             node_seed, new_error_start, aon_base_path, aon_filename, activation_constant, gd_iterations, cost_fxn,
             allow_summary_print, summary_print_interval, batch_method, gd_method, conv_method, lr_method, momentum_weight,
             non_neg_coeffs, iteration).run()








